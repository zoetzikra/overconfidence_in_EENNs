from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import os
import math
import time

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
from laplace import estimate_variance_efficient
import random
import sys

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
def calc_ensemble_logits(logits, flop_weights):
    ens_logits = torch.zeros_like(logits)
    ens_logits[0,:,:] = logits[0,:,:].clone()
    
    p = flop_weights[0]
    summ = p*logits[0,:,:].clone()

    w = p
    for i in range(1,logits.shape[0]):
        p = flop_weights[i]
        summ += p*logits[i,:,:].clone()
        w += p
        ens_logits[i,:,:] = summ / w

    return ens_logits
        
def Entropy(p):
    # Calculates the sample entropies for a batch of output softmax values
    '''
        p: m * n * c
        m: Exits
        n: Samples
        c: Classes
    '''
    Ex = -1*torch.sum(p*torch.log(p), dim=2)
    return Ex
    
def accuracy(output, target, topk=(1,)):
    """Computes the precor@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    
def calc_bins(confs, corrs):
    # confs and corrs are numpy arrays
    # Assign each prediction to a bin
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(confs, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(confs[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (corrs[binned==bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (confs[binned==bin]).sum() / bin_sizes[bin]

    return bins, bin_accs, bin_confs, bin_sizes
    
def calculate_ECE(confs, corrs):
    # confs and corrs are numpy arrays
    ECE = 0
    bins, bin_accs, bin_confs, bin_sizes = calc_bins(confs, corrs)
    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif

    return ECE

def calculate_signed_ECE(confidences, corrects, n_bins=15):
    """
    Calculate the signed Expected Calibration Error (ECE).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = corrects[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += prop_in_bin * (avg_confidence_in_bin - accuracy_in_bin)
    
    return ece

def dynamic_evaluate(model, test_loader, val_loader, args, prints = False):
    tester = Tester(model, args)
    tester.confidence_thresholds = [0.7, 0.8, 0.9, 0.95]  # FOR THE PROBES # NOT USED IN THE BUDGETED CLASSIFICATION EXPERIMENTS
        
    # Expected computational cost of each block for the whole dataset             
    flops = torch.load(os.path.join(args.save, 'flops.pth'))
    print(flops)
    flop_weights = np.array(flops)/np.array(flops)[-1] #.sum()
    print(flop_weights)
        
    ############ Set file naming strings based on options selected ############
    fname_ending = ''
    fname_ending += '_mie' if args.MIE else ''
    fname_ending += '_opttemp' if args.optimize_temperature else ''
    fname_ending += '_optvar' if args.optimize_var0 else ''
    
    ###########################################################################

    '''Addition for uncertainty and exit point computation'''
    # Initialize matrices for tracking uncertainties, exit points, and correctness
    n_test = len(test_loader.sampler)
    uncertainties = torch.zeros(args.nBlocks, n_test)
    exit_points = torch.zeros(n_test, dtype=torch.long)
    correct_predictions = torch.zeros(n_test, dtype=torch.bool)

    # Optimize the temperature scaling parameters
    if args.optimize_temperature:
        print('******* Optimizing temperatures scales ********')
        tester.args.laplace_temperature = [1.0 for i in range(args.nBlocks)]
        temp_grid = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]
    else:
        temp_grid = [args.laplace_temperature]
    if args.optimize_var0:
        print('******* Optimizing Laplace prior variance ********')
        var_grid = [0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0]
    else:
        var_grid = [args.var0]
    max_count = len(var_grid)*len(temp_grid)
    if max_count > 1:
        count = 1
        if not args.MIE:
            results = torch.zeros(args.nBlocks, len(temp_grid), len(var_grid))
            for j in range(len(temp_grid)):
                for i in range(len(var_grid)):
                    temp = temp_grid[j]
                    var0 = var_grid[i]
                    print('Optimizing setup {}/{}'.format(count, max_count))
                    tester.args.laplace_temperature = [temp for t in range(args.nBlocks)]
                    blockPrint()
                    if not args.laplace:
                        val_pred_o, val_target_o = tester.calc_logit(val_loader, temperature=[temp for t in range(args.nBlocks)])
                    else:
                        val_pred_o, val_target_o, _ = tester.calc_la_logit(val_loader, [var0])
                    enablePrint()
                    
                    for block in range(args.nBlocks):
                        nlpd_o = nn.functional.nll_loss(torch.log(val_pred_o[block,:,:]), val_target_o)
                        results[block,j,i] = -1*nlpd_o
                    count += 1
            optimized_vars, optimized_temps = [], []
            for block in range(args.nBlocks):
                max_ind = (results[block,:,:]==torch.max(results[block,:,:])).nonzero().squeeze()
                temp_o = temp_grid[max_ind[0]]
                var_o = var_grid[max_ind[1]]
                optimized_temps.append(temp_o)
                optimized_vars.append(var_o)
                print('For block {}, best temperature is {} and best var0 is {}'.format(block+1, temp_o, var_o))
                print()
        else:
            optimized_temps, optimized_vars = [0 for t in range(args.nBlocks)],[0 for t in range(args.nBlocks)]
            current_temps = [0 for t in range(args.nBlocks)]
            current_vars = [0 for t in range(args.nBlocks)]
            for exit in range(args.nBlocks):
                count = 1
                results = torch.zeros(len(temp_grid), len(var_grid))
                print('Optimizing for exit {}'.format(exit+1))
                for j in range(len(temp_grid)):
                    for i in range(len(var_grid)):
                        temp = temp_grid[j]
                        var0 = var_grid[i]
                        print('Optimizing setup {}/{}'.format(count, max_count))
                        current_temps[0:exit+1] = optimized_temps[0:exit+1]
                        current_temps[exit] = temp
                        current_vars[0:exit+1] = optimized_vars[0:exit+1]
                        current_vars[exit] = var0
                        tester.args.laplace_temperature = current_temps
                        blockPrint()
                        if not args.laplace:
                            val_pred_o, val_target_o = tester.calc_logit(val_loader, temperature=current_temps, until=exit+1)
                        else:
                            val_pred_o, val_target_o, _ = tester.calc_la_logit(val_loader, current_vars, until=exit+1)
                        enablePrint()
                        val_pred = calc_ensemble_logits(val_pred_o, flop_weights)
                        
                        nlpd_o = nn.functional.nll_loss(torch.log(val_pred[exit,:,:]), val_target_o)
                        results[j,i] = -1*nlpd_o
                        count += 1
                        
                max_ind = (results==torch.max(results)).nonzero().squeeze()
                temp_o = temp_grid[max_ind[0]]
                var_o = var_grid[max_ind[1]]
                optimized_temps[exit] = temp_o
                optimized_vars[exit] = var_o
                print('For block {}, best temperature is {} and best var0 is {}'.format(exit+1, temp_o, var_o))
                print()

        
        tester.args.laplace_temperature = optimized_temps
        args.laplace_temperature = optimized_temps
        vanilla_temps = optimized_temps
        args.var0 = optimized_vars
        print(optimized_temps)
        print(optimized_vars)
    else:
        vanilla_temps = None
        args.var0 = [args.var0]
        tester.args.laplace_temperature = [args.laplace_temperature]
        
    # Calculate validation and test predictions
    '''
    val_pred, test_pred are softmax outputs, shape (n_blocks, n_samples, n_classes)
    val_var, test_var are predicted class variances, shape (n_blocks, n_samples)
    '''
    if not args.laplace:
        # Use softmax confidence-based evaluation
        filename = os.path.join(args.save, 'actual_dynamic_softmax%s.txt' % (fname_ending))
        
        # Calculate confidences using softmax
        val_logits, val_confidences, val_targets = tester.calc_softmax_confidence(val_loader)
        test_logits, test_confidences, test_targets = tester.calc_softmax_confidence(test_loader)
        
        # Convert logits to predictions using softmax
        val_pred = torch.nn.functional.softmax(val_logits, dim=2)
        test_pred = torch.nn.functional.softmax(test_logits, dim=2)
        val_target = val_targets
        test_target = test_targets

        #THIS EXTRA ADDITION WAS MISSING AT FIRST BUT IT WAS PRESENT IN THE INITIAL VERSION
        # Calculate validation and test set accuracies for each block (add this back)
        _, argmax_val = val_pred.max(dim=2, keepdim=False)
        maxpred_test, argmax_test = test_pred.max(dim=2, keepdim=False)
        print('Val acc      Test acc')
        for e in range(val_pred.shape[0]):
            val_acc = (argmax_val[e,:] == val_target).sum()/val_pred.shape[1]
            test_acc = (argmax_test[e,:] == test_target).sum()/test_pred.shape[1]
            print('{:.3f}       {:.3f}'.format(val_acc, test_acc))
        print('')

        # Initialize matrices for tracking
        n_samples = len(test_targets)
        n_exits = args.nBlocks
        # the next 3 are new:
        uncertainties = torch.zeros((n_exits, n_samples))
        exit_points = torch.zeros(n_samples, dtype=torch.long)
        correct_predictions = torch.zeros(n_samples, dtype=torch.bool)
        
        # Calculate base correctness for each exit
        _, predictions = test_pred.max(dim=2)
        correctness = torch.zeros((n_exits, n_samples))
        for i in range(n_exits):
            correctness[i] = (predictions[i] == test_targets).float()

        # Open file for detailed results
        with open(os.path.join(args.save, 'confidence_correctness.txt'), 'w') as f:
            # Write header
            f.write('Sample\tBudget')
            for i in range(n_exits):
                f.write(f'\tConf_Exit_{i+1}')
            for i in range(n_exits):
                f.write(f'\tCorrect_Exit_{i+1}')
            f.write('\tActual_Exit\tActual_FLOPs\n')
                
            # Iterate over different budget levels
            for p in range(1, 40):
                print(f"\n{'*'*20} Budget Level {p}/39 {'*'*20}")

                _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
                probs = torch.exp(torch.log(_p) * torch.arange(1, args.nBlocks+1))
                probs /= probs.sum()
                
                # Find dynamic thresholds
                acc_val, _, T = tester.dynamic_find_threshold(val_pred, val_target, val_confidences, probs, flops)
                
                # Print dynamic thresholds for this budget level (new addition)
                print(f"\nDynamic Thresholds and Accuracy for Budget Level {p}/39:")
                print(f"Dynamic Val Accuracy: {acc_val:.3f}")  # Accuracy with early-exit strategy
                print("Exit thresholds:")
                for i, threshold in enumerate(T):
                    print(f"Exit {i+1}: {threshold:.3f}")

                # For each sample
                for i in range(n_samples):
                    f.write(f'{i}\t{p}')
                    # Write confidences
                    for j in range(n_exits):
                        f.write(f'\t{test_confidences[j][i]:.4f}')
                    # Write correctness
                    for j in range(n_exits):
                        f.write(f'\t{int(correctness[j][i])}')
                    
                    # Determine exit point using dynamic thresholds
                    actual_exit = None
                    for k in range(n_exits):
                        if test_confidences[k][i] >= T[k]:
                            actual_exit = k
                            break
                    if actual_exit is None:
                        actual_exit = n_exits - 1
                    
                    # Calculate FLOPs and record exit point
                    actual_flops = sum(flops[:actual_exit + 1])
                    exit_points[i] = actual_exit
                    correct_predictions[i] = correctness[actual_exit][i]
                    uncertainties[:, i] = 1 - test_confidences[:, i]
                    
                    f.write(f'\t{actual_exit + 1}\t{actual_flops:.0f}\n')
                
                # Calculate and evaluate with these thresholds
                acc_test, exp_flops, nlpd, ECE, acc5 = tester.dynamic_eval_threshold(
                    test_pred, test_target, flops, T, test_confidences, p)
                
                # Calculate exit distribution for this budget level
                exit_counts = torch.zeros(n_exits)
                for i in range(n_samples):
                    exit_counts[int(exit_points[i])] += 1
                exit_distribution = exit_counts / n_samples * 100
                
                print(f"\nBudget level {p}/39 (FLOPs: {exp_flops/1e6:.2f}M)")
                print("Exit point distribution:")
                for i, pct in enumerate(exit_distribution):
                    print(f"Exit {i+1}: {pct:.1f}%")
                print(f"Accuracy: {acc_test:.3f}, ECE: {ECE:.3f}, NLPD: {nlpd:.3f}")

        # Save results
        results_dir = os.path.join(args.save, 'test_results')
        os.makedirs(results_dir, exist_ok=True)
        np.save(os.path.join(results_dir, 'uncertainties.npy'), uncertainties.cpu().numpy())
        np.save(os.path.join(results_dir, 'exit_points.npy'), exit_points.cpu().numpy())
        np.save(os.path.join(results_dir, 'correct_predictions.npy'), correct_predictions.cpu().numpy())
        
        # Save final summary
        with open(os.path.join(results_dir, 'test_results.txt'), 'w') as f:
            f.write('Sample\tExit_Point\tCorrect\tConfidences\n')
            for i in range(n_samples):
                confidences_str = '\t'.join([f'{c:.4f}' for c in test_confidences[:, i]])
                f.write(f'{i}\t{exit_points[i]}\t{int(correct_predictions[i])}\t{confidences_str}\n')


        # if args.fixed_threshold_eval: 
        #     print("\nEvaluating with fixed thresholds for visualization...")
        #     print(f"Using classifier threshold: {args.classifier_threshold}")
            
        #     # Fixed thresholds setup
        #     probe_thresholds = np.linspace(0.1, 1.0, 10)  # 10 different thresholds for probes
            
        #     results = []
        #     n_samples = len(test_loader.dataset)
        #     n_exits = args.nBlocks

        #     # For each probe threshold
        #     for probe_thresh in probe_thresholds:
        #         print(f"\nEvaluating with probe threshold: {probe_thresh:.2f}")
        #         exit_points = torch.zeros(n_samples)
        #         correct_predictions = torch.zeros(n_samples, dtype=torch.bool)
        #         sample_idx = 0

        #         for i, (input, target) in enumerate(test_loader):
        #             input = input.cuda()
        #             target = target.cuda()
        #             batch_size = input.size(0)

        #             # Get classifier and probe confidences separately
        #             classifier_outputs, classifier_confidences = model.module.compute_confidence_scores_classifier(input)
        #             probe_outputs, probe_confidences = model.module.compute_confidence_scores_probe(input)


        #             # For each sample in batch
        #             for j in range(batch_size):
        #                 # Find earliest exit that meets either threshold
        #                 exit_found = False
        #                 for exit in range(n_exits):
        #                     cls_conf = classifier_confidences[exit][j]
        #                     probe_conf = probe_confidences[exit][j]

        #                     if cls_conf >= args.classifier_threshold:
        #                         exit_points[sample_idx + j] = exit
        #                         # But ALWAYS use classifier output for the actual prediction
        #                         pred = classifier_outputs[exit][j].argmax()  # Use classifier prediction
        #                         correct_predictions[sample_idx + j] = (pred == target[j])
        #                         exit_found = True
        #                         break
                            
        #                     elif probe_conf >= probe_thresh:
        #                         exit_points[sample_idx + j] = exit
        #                         pred = probe_outputs[exit][j].argmax()  # Use probe prediction
        #                         correct_predictions[sample_idx + j] = (pred == target[j])
        #                         exit_found = True
        #                         break

        #                 # If no exit found, use last exit with classifier output
        #                 if not exit_found:
        #                     exit_points[sample_idx + j] = n_exits - 1
        #                     pred = classifier_outputs[-1][j].argmax()
        #                     correct_predictions[sample_idx + j] = (pred == target[j])

        #             sample_idx += batch_size

        #         # Calculate metrics
        #         avg_exit = exit_points.float().mean().item() + 1  # +1 for 1-based indexing
        #         accuracy = correct_predictions.float().mean().item() * 100

        #         results.append({
        #             'probe_threshold': probe_thresh,
        #             'avg_exit': avg_exit,
        #             'accuracy': accuracy,
        #             'classifier_threshold': args.classifier_threshold
        #         })

        #         print(f"Average Exit: {avg_exit:.2f}, Accuracy: {accuracy:.2f}%")

        #     # Save results for plotting
        #     results_path = os.path.join(args.save, 'fixed_threshold_results.json')
        #     import json
        #     with open(results_path, 'w') as f:
        #         json.dump(results, f)

        return acc_test, nlpd, ECE, acc5, exp_flops

        
    else:
        if args.optimize_temperature and args.optimize_var0:
            filename = os.path.join(args.save, 'dynamic_la_mc%03d%s.txt' % (args.n_mc_samples, fname_ending))
        elif args.optimize_temperature:
            filename = os.path.join(args.save, 'dynamic_la_priorvar%01.4f_mc%03d%s.txt' % (args.var0[0], args.n_mc_samples, fname_ending))
        elif args.optimize_var0:
            filename = os.path.join(args.save, 'dynamic_la_mc%03d_temp%01.2f%s.txt' % (args.n_mc_samples, args.laplace_temperature[0], fname_ending))
        else:
            filename = os.path.join(args.save, 'dynamic_la_priorvar%01.4f_mc%03d_temp%01.2f%s.txt' % (args.var0[0], args.n_mc_samples, args.laplace_temperature[0], fname_ending))

        val_pred, val_target, var0 = tester.calc_la_logit(val_loader, args.var0)
        test_pred, test_target, var0 = tester.calc_la_logit(test_loader, args.var0)
      
    if args.MIE:
        val_pred = calc_ensemble_logits(val_pred, flop_weights)
        test_pred = calc_ensemble_logits(test_pred, flop_weights)          
                
    # Calculate validation and test set accuracies for each block
    maxpred_val, argmax_val = val_pred.max(dim=2, keepdim=False) #predicted class confidences
    maxpred_test, argmax_test = test_pred.max(dim=2, keepdim=False)
    print('Val acc      Test acc')
    for e in range(val_pred.shape[0]):
        val_acc = (argmax_val[e,:] == val_target).sum()/val_pred.shape[1]
        test_acc = (argmax_test[e,:] == test_target).sum()/test_pred.shape[1]
        print('{:.3f}       {:.3f}'.format(val_acc, test_acc))
    print('')
    
    with open(filename, 'w') as fout:
        exit_distributions = []  # List to store exit distributions for each budget
        for p in range(1, 40): # Loop over 40 different computational budget levels
            print("*********************")
            exit_counts = torch.zeros(args.nBlocks)  # Counter for each exit point
            total_samples = 0

            _p = torch.FloatTensor(1).fill_(p * 1.0 / 20) # 'Heaviness level' of the current computational budget
            probs = torch.exp(torch.log(_p) * torch.arange(1, args.nBlocks+1)) # Calculate proportions of computation for each DNN block
            probs /= probs.sum() # normalize
            
            val_t_metric_values, _ = val_pred.max(dim=2, keepdim=False) #predicted class confidences
            test_t_metric_values, _ = test_pred.max(dim=2, keepdim=False)
        
            # Find thresholds to determine which block handles each sample
            acc_val, _, T = tester.dynamic_find_threshold(val_pred, val_target, val_t_metric_values, probs, flops)
                
            # Calculate accuracy, expected computational cost, nlpd and ECE given thresholds in T
            acc_test, exp_flops, nlpd, ECE, acc5 = tester.dynamic_eval_threshold(test_pred, test_target, flops, T, test_t_metric_values, p)
                
            print('valid acc: {:.3f}, test acc: {:.3f}, test top5 acc: {:.3f} nlpd: {:.3f}, ECE: {:.3f}, test flops: {:.2f}'.format(acc_val, acc_test, acc5, nlpd, ECE, exp_flops / 1e6))

            # During evaluation of each sample
            for i in range(args.nBlocks):
                t_metric_values = test_t_metric_values[i]
                exit_mask = t_metric_values >= T[i] if i == 0 else (t_metric_values >= T[i]) & (test_t_metric_values[i-1] < T[i-1])
                # For first exit (i=0), only check if current metric exceeds threshold
                if i == 0: # For first exit, only check if metric value exceeds threshold No need to check previous thresholds since this is the first exit
                    exit_mask = (t_metric_values >= T[i])  # True if sample should exit at first block
                else: # For later exits (i>0), check two conditions: 1. Current metric exceeds current threshold 2. Previous metric was below previous threshold
                    current_exceeds_threshold = t_metric_values >= T[i]
                    prev_below_threshold = test_t_metric_values[i-1] < T[i-1]
                    exit_mask = current_exceeds_threshold & prev_below_threshold
                    
                exit_counts[i] = exit_mask.sum().item()

                '''Addition for uncertainty and exit point computation'''
                # Record uncertainties, exit points, and correctness
                for sample in range(test_pred.shape[1]):
                    if exit_mask[sample]:
                        exit_points[sample] = i
                        pred = test_pred[i, sample].argmax().item()
                        correct_predictions[sample] = (pred == test_target[sample].item())
                        uncertainties[i, sample] = 1 - test_t_metric_values[i, sample]
                            
            # Calculate distribution
            total_samples = len(test_target)
            exit_distribution = exit_counts / total_samples
            exit_distributions.append(exit_distribution)
            
            print(f"\nBudget level {p}/39 (FLOPs: {exp_flops/1e6:.2f}M)")
            print("Exit point distribution:")
            for i, pct in enumerate(exit_distribution):
                print(f"Exit {i+1}: {pct*100:.1f}%")

            fout.write('{}\t{}\t{}\t{}\t{}\n'.format(acc_test, nlpd, ECE, acc5, exp_flops.item()))       

    '''Addition for uncertainty and exit point computation'''
    # Save results
    results_dir = os.path.join(args.save, 'test_results')
    os.makedirs(results_dir, exist_ok=True)
    # Save matrices
    np.save(os.path.join(results_dir, 'uncertainties.npy'), uncertainties.cpu().numpy())
    np.save(os.path.join(results_dir, 'exit_points.npy'), exit_points.cpu().numpy())
    np.save(os.path.join(results_dir, 'correct_predictions.npy'), correct_predictions.cpu().numpy())
    # Save human-readable summary
    with open(os.path.join(results_dir, 'test_results.txt'), 'w') as f:
        f.write('Sample\tExit_Point\tCorrect\tUncertainties\n')
        for i in range(n_test):
            uncertainties_str = '\t'.join([f'{u:.4f}' for u in uncertainties[:, i]])
            f.write(f'{i}\t{exit_points[i]}\t{int(correct_predictions[i])}\t{uncertainties_str}\n')

class Tester(object):
    def __init__(self, model, args=None):
        self.args = args
        self.model = model
        self.softmax = nn.Softmax(dim=1).cuda()

    def calc_softmax_confidence(self, dataloader):
        """
        This function collects:
        - logits from each exit
        - confidence scores from each exit
        - targets
        """
        
        self.model.eval()
        n_exit = self.args.nBlocks
        
        logits = [[] for _ in range(n_exit)]
        confidences = [[] for _ in range(n_exit)]
        targets = []
        
        start_time = time.time()
        for i, (input, target) in enumerate(dataloader):
            targets.append(target)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input).cuda()
                
                # Choose confidence computation method based on whether probes are being used
                if self.args.evaluate_probe_from:
                    output, confs = self.model.module.compute_combined_confidence_scores(input_var)
                    confidence_type = "Combined"
                elif "model_best_probe_acc.pth.tar" in self.args.evaluate_from:
                    output, confs = self.model.module.compute_confidence_scores_probe(input_var)
                    confidence_type = "Probe"
                else:
                    output, confs = self.model.module.compute_confidence_scores_classifier(input_var)
                    confidence_type = "Classifier"
            
                if not isinstance(output, list):
                    output = [output]
                    
                # Store logits and confidences
                for b in range(n_exit):
                    logits[b].append(output[b])
                    confidences[b].append(confs[b])

            if i % self.args.print_freq == 0:
                print(f'Generate {confidence_type} Confidence: [{i}/{len(dataloader)}]')
    
        # Format outputs similar to calc_logit
        for b in range(n_exit):
            logits[b] = torch.cat(logits[b], dim=0)
            confidences[b] = torch.cat(confidences[b], dim=0)

        # Create tensors of appropriate size
        size = (n_exit, logits[0].size(0), logits[0].size(1))
        ts_logits = torch.Tensor().resize_(size).zero_()
        ts_confidences = torch.Tensor(n_exit, logits[0].size(0)).zero_()
        
        for b in range(n_exit):
            ts_logits[b].copy_(logits[b])
            ts_confidences[b].copy_(confidences[b])

        targets = torch.cat(targets, dim=0)
        ts_targets = torch.Tensor().resize_(size[1]).copy_(targets)
        
        print(f'{confidence_type} confidence calculation time: {time.time() - start_time}')

        return ts_logits, ts_confidences, ts_targets

    def calc_logit(self, dataloader, temperature=None, until=None):
        self.model.eval()
        if until is not None:
            n_exit = until
        else:
            n_exit = self.args.nBlocks
        logits = [[] for _ in range(n_exit)]
        targets = []
        start_time = time.time()
        for i, (input, target) in enumerate(dataloader):
            targets.append(target)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input).cuda()
                #input_var = torch.autograd.Variable(input)
                if until is not None:
                    output, phi = self.model.module.predict_until(input_var, until)
                else:
                    output, phi = self.model.module.predict(input_var)
                #output = self.model(input_var)
                if not isinstance(output, list):
                    output = [output]
                for b in range(n_exit):
                    if temperature is not None:
                        _t = self.softmax(output[b]/temperature[b])
                    else:
                        _t = self.softmax(output[b])

                    logits[b].append(_t) 

            if i % self.args.print_freq == 0:
                print('Generate Logit: [{0}/{1}]'.format(i, len(dataloader)))

        for b in range(n_exit):
            logits[b] = torch.cat(logits[b], dim=0)

        size = (n_exit, logits[0].size(0), logits[0].size(1))
        ts_logits = torch.Tensor().resize_(size).zero_()
        for b in range(n_exit):
            ts_logits[b].copy_(logits[b])

        targets = torch.cat(targets, dim=0)
        ts_targets = torch.Tensor().resize_(size[1]).copy_(targets)
        print('Logits calculation time: {}'.format(time.time() - start_time))

        return ts_logits, targets
        
    def calc_la_logit(self, dataloader, var0, until=None):
        self.model.eval()
        if until is not None:
            n_exit = until
        else:
            n_exit = self.args.nBlocks

        var0 = [torch.tensor(var0[j]).float().cuda() for j in range(len(var0))]
        M_W, U, V = list(np.load(os.path.join(self.args.save, "effL_llla.npy"), allow_pickle=True))
        
        M_W = [torch.from_numpy(M_W[j]).cuda() for j in range(n_exit)] # shape in features x out features (n_classes)
        U = [torch.from_numpy(U[j]).cuda() for j in range(n_exit)]  # n_classes x n_classes
        V = [torch.from_numpy(V[j]).cuda() for j in range(n_exit)]  # n_features x n_features
        M_W, U, V = estimate_variance_efficient(var0, [M_W, U, V])
        n_classes = U[0].shape[0]

        Lz = [[] for j in range(len(U))]
        L = [torch.linalg.cholesky(U[j]) for j in range(len(U))]
        for i in range(self.args.n_mc_samples):
            z = torch.randn(n_classes).cuda()
            for j in range(len(U)):
                Lz[j].append((L[j] @ z).squeeze())

        logits = [[] for _ in range(n_exit)]
        targets = []
        start_time = time.time()
        for i, (input, target) in enumerate(dataloader):
            targets.append(target)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input).cuda()
                if until is not None:
                    output, phi = self.model.module.predict_until(input_var, until) # Calculate model output and mean feature of the image (phi)
                else:
                    output, phi = self.model.module.predict(input_var)
                # output shape: n_batch x n_classes (64 x 100)
                # phi shape: n_batch x n_features (64 x 128)

                phi = [torch.cat((phi[j], torch.ones_like(phi[j][:,0]).unsqueeze(-1)),dim=-1) for j in range(len(phi))]
                output1 = [phi[j] @ M_W[j] for j in range(len(phi))]
                s = [torch.diag(phi[j] @ V[j] @ phi[j].t()).view(-1, 1) for j in range(len(phi))]

                output_mc = []
                for j in range(len(phi)):
                    py_ = 0
                    for mc_sample in range(self.args.n_mc_samples):
                        if self.args.optimize_temperature:
                            py = (output1[j] + torch.sqrt(s[j])*Lz[j][mc_sample].unsqueeze(0)) / self.args.laplace_temperature[j]

                        else:
                            py = (output1[j] + torch.sqrt(s[j])*Lz[j][mc_sample].unsqueeze(0)) / self.args.laplace_temperature[0]
                        py_ += self.softmax(py) # HERE SOFTMAX IS APPLIED TO EACH MC SAMPLE
                    py_ /= self.args.n_mc_samples
                    
                    output_mc.append(py_)
                if not isinstance(output_mc, list):
                    output_mc = [output_mc]
                for b in range(n_exit):
                    logits[b].append(output_mc[b])

            if i % self.args.print_freq == 0: 
                print('Generate Logit: [{0}/{1}]'.format(i, len(dataloader)))

        for b in range(n_exit):
            logits[b] = torch.cat(logits[b], dim=0)

        size = (n_exit, logits[0].size(0), logits[0].size(1))
        ts_logits = torch.Tensor().resize_(size).zero_()
        for b in range(n_exit):
            ts_logits[b].copy_(logits[b])

        targets = torch.cat(targets, dim=0)
        ts_targets = torch.Tensor().resize_(size[1]).copy_(targets)
        print('Laplace logits calculation time: {}'.format(time.time() - start_time))
        
        return ts_logits, targets, var0 # NOT REALLY LOGITS BUT PREDICTIONS
        

    def dynamic_find_threshold(self, logits, targets, t_metric_values, p, flops):
        """
            logits: m * n * c
            m: Exits
            n: Samples
            c: Classes
            
            t_metric_values: m * n
        """
        # Define whether uncertainty is descending or ascending as threshold metric value increases
        descend = True # This allows using other metrics as threshold metric to exit samples
            
        n_exit, n_sample, c = logits.size()
        _, argmax_preds = logits.max(dim=2, keepdim=False) # Predicted class index for each stage and sample
        _, sorted_idx = t_metric_values.sort(dim=1, descending=descend) # Sort threshold metric values for each stage

        filtered = torch.zeros(n_sample)
        
        # Initialize thresholds
        T = torch.Tensor(n_exit).fill_(1e8) if descend else torch.Tensor(n_exit).fill_(-1e8)

        for k in range(n_exit - 1):
            acc, count = 0.0, 0
            out_n = math.floor(n_sample * p[k]) # Number of samples that should be exited at stage k
            for i in range(n_sample):
                ori_idx = sorted_idx[k][i] # Original index of the sorted sample
                if filtered[ori_idx] == 0: # Check if the sample has already been exited from an earlier stage
                    count += 1 # Add 1 to the count of samples exited at stage k
                    if count == out_n:
                        T[k] = t_metric_values[k][ori_idx] # Set threshold k to value of the last sample exited at exit k
                        break
            #Add 1 to filtered in locations of samples that were exited at stage k
            if descend:
                filtered.add_(t_metric_values[k].ge(T[k]).type_as(filtered))
            else:
                filtered.add_(t_metric_values[k].le(T[k]).type_as(filtered))

        # accept all of the samples at the last stage
        T[n_exit -1] = -1e8 if descend else 1e8

        acc_rec, exp = torch.zeros(n_exit), torch.zeros(n_exit)
        acc, expected_flops = 0, 0 # Initialize accuracy and expected cumulative computational cost
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_exit):
                t_ki = t_metric_values[k][i].item() #current threshold metric value
                exit_test = t_ki >= T[k] if descend else t_ki <= T[k]
                if exit_test: # force the sample to exit at k
                    if int(gold_label.item()) == int(argmax_preds[k][i].item()): # check if prediction was correct
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all = 0

        for k in range(n_exit):
            _t = 1.0 * exp[k] / n_sample # The fraction of samples that were exited at stage k
            expected_flops += _t * flops[k] # Add the computational cost from usage of stage k
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops, T


    def dynamic_eval_threshold(self, logits, targets, flops, T, t_metric_values, p):
        # Define whether uncertainty is descending or ascending as threshold metric value increases
        descend = True # This allows using other metrics as threshold metric to exit samples
        
        n_exit, n_sample, n_class = logits.size()
        maxpreds, argmax_preds = logits.max(dim=2, keepdim=False) # predicted class indexes

        acc_rec, exp = torch.zeros(n_exit), torch.zeros(n_exit)
        acc, expected_flops = 0, 0
        nlpd = 0 # Initialize cumulative nlpd
        final_confs = torch.zeros(n_sample) #Tensor for saving confidences for each sample based on which block was used
        final_corrs = torch.zeros(n_sample) #Prediction correctness of final preds
        final_logits = torch.zeros(n_sample, n_class)
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_exit):
                t_ki = t_metric_values[k][i].item() #current threshold metric value
                exit_test = t_ki >= T[k] if descend else t_ki <= T[k]
                if exit_test: # force the sample to exit at k
                    _g = int(gold_label.item())
                    _pred = int(argmax_preds[k][i].item())
                    if _g == _pred:
                        final_corrs[i] = 1
                        acc += 1
                        acc_rec[k] += 1
                    final_confs[i] = maxpreds[k][i]
                    exp[k] += 1
                    nlpd += -1*logits[k,i,_g].log()
                    final_logits[i,:] = logits[k,i,:]

                    break
        acc_all, sample_all = 0, 0
        for k in range(n_exit):
            _t = exp[k] * 1.0 / n_sample # The fraction of samples that were exited at stage k
            sample_all += exp[k]
            expected_flops += _t * flops[k] # Add the computational cost from usage of stage k
            acc_all += acc_rec[k]
            
        ECE = calculate_ECE(final_confs.numpy(), final_corrs.numpy())
            
        prec5 = accuracy(final_logits, targets, topk=(5,))

        return acc * 100.0 / n_sample, expected_flops, nlpd / n_sample, ECE, prec5[0]
        

    # NOT USED IN THE BUDGETED CLASSIFICATION EXPERIMENTS
    def dynamic_eval_with_softmax(self, logits, targets, flops, confidence_thresholds):
        """
        Dynamic evaluation using maximum softmax probability as confidence measure
        Args:
            logits: predictions from each exit (n_exits x batch_size x n_classes)
            targets: ground truth labels
            flops: computational cost for each exit
            confidence_thresholds: list of confidence thresholds for each exit

        Characteristics:
        - Uses maximum softmax probability as confidence measure
        - Each exit has its own confidence threshold
        - Early exits when confidence exceeds threshold
        - Tracks accuracy and ECE
        """
        n_exit, n_sample, n_class = logits.size()
        acc_rec, exp = torch.zeros(n_exit), torch.zeros(n_exit)
        acc, expected_flops = 0, 0
        
        # Track exits
        exit_counts = torch.zeros(n_exit)
        
        final_preds = torch.zeros(n_sample)
        final_confs = torch.zeros(n_sample)
        final_corrs = torch.zeros(n_sample)  # Added for ECE calculation
        final_logits = torch.zeros(n_sample, n_class)  # Added for top-5 accuracy
    
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_exit):
                # 1. Compute confidence as maximum softmax probability
                current_logits = logits[k, i].unsqueeze(0)
                confidence = self.model.module.compute_max_softmax_confidence(current_logits)
                
                # Debugging: Print confidence values
                print(f"Sample {i}, Exit {k+1}, Confidence: {confidence.item():.4f}")

                # 2. Check if confidence meets threshold
                if confidence >= confidence_thresholds[k]:
                    pred = current_logits.max(1)[1]
                    final_preds[i] = pred
                    final_confs[i] = confidence
                    final_logits[i,:] = logits[k,i,:]
                    
                    # 3. Update metrics
                    if pred == gold_label:
                        final_corrs[i] = 1  # For ECE calculation
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    expected_flops += flops[k]
                    break
                    
                # 4. If we've reached the last exit, use it regardless of confidence
                if k == n_exit - 1:
                    pred = current_logits.max(1)[1]
                    final_preds[i] = pred
                    final_confs[i] = confidence
                    final_logits[i,:] = logits[k,i,:]

                    if pred == gold_label:
                        final_corrs[i] = 1 
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    expected_flops += flops[k]
        
        # Calculate ECE
        ECE = calculate_ECE(final_confs.numpy(), final_corrs.numpy())
        # Calculate signed ECE
        signed_ECE = calculate_signed_ECE(final_confs.numpy(), final_corrs.numpy())
        # Calculate top-5 accuracy
        prec5 = accuracy(final_logits, targets, topk=(5,))
        # Calculate the nlpd, Convert indices to long tensor
        idx = torch.arange(n_sample, dtype=torch.long)
        targets_long = targets.long()
        nlpd = -1 * torch.sum(final_logits[idx, targets_long])
        
        return acc * 100.0 / n_sample, expected_flops, nlpd, ECE, signed_ECE, prec5[0]
