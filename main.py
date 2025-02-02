#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import math
import time
import shutil
import numpy as np

from dataloader import get_dataloaders
from args import arg_parser
from adaptive_inference import dynamic_evaluate
import models
from op_counter import measure_model
#
from laplace import get_hessian_efficient

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system') 

args = arg_parser.parse_args()

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)

if args.use_valid:
    args.splits = ['train', 'val', 'test']
else:
    args.splits = ['train', 'val']

if args.data == 'cifar10':
    args.num_classes = 10
elif args.data == 'cifar100':
    args.num_classes = 100
elif args.data == 'caltech256':
    args.num_classes = 257
else:
    args.num_classes = 1000

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms

torch.manual_seed(args.seed)

def main():

    global args
    best_prec1, best_epoch = 0.0, 0

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.data.startswith('cifar'):
        IM_SIZE = 32
    else:
        IM_SIZE = 224

    model = getattr(models, args.arch)(args)
    n_flops, n_params = measure_model(model, IM_SIZE, IM_SIZE)
    args.n_flops = n_flops    
    torch.save(n_flops, os.path.join(args.save, 'flops.pth'))
    del(model)
        
        
    model = getattr(models, args.arch)(args)
    model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    if args.resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    train_loader, val_loader, test_loader = get_dataloaders(args)

    if args.evalmode is not None:
        state_dict = torch.load(args.evaluate_from)['state_dict']
        model.load_state_dict(state_dict)

        # Load probe checkpoint if provided
        if args.confidence_type == "probe" or args.confidence_type == "combined": # if the confidence type is classifier, we don't need to load the probe checkpoint even if it is provided
            probe_state = torch.load(args.evaluate_probe_from)['state_dict']
            current_state = model.state_dict()
            # Only load probe weights
            for k, v in probe_state.items():
                if 'probe' in k:
                    current_state[k] = v
            model.load_state_dict(current_state)

        if args.evalmode == 'anytime':
            validate(test_loader, model, criterion)
        else:
            dynamic_evaluate(model, test_loader, val_loader, args)
        return

    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_prec1'
              '\tval_prec1\ttrain_prec5\tval_prec5']

    # if not args.compute_only_laplace:
    #     initial_time = time.time()
    #     # Track last saved checkpoint number
    #     last_checkpoint_num = args.epochs - 1  # This will be the last classifier checkpoint number
        
    #     for epoch in range(args.start_epoch, args.epochs):
        
    #         model_filename = 'checkpoint_%03d.pth.tar' % epoch
    #         # Train and validate main model 
    #         train_loss, train_prec1, train_prec5, lr = train(train_loader, model, criterion, optimizer, epoch)
    #         val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion)
    #         # Update scores
    #         scores.append(('{}\t{:.3f}' + '\t{:.4f}' * 6).format(epoch, lr, train_loss, val_loss, train_prec1, val_prec1, train_prec5, val_prec5))
            
    #         is_best = val_prec1 > best_prec1
    #         if is_best:
    #             best_prec1 = val_prec1
    #             best_epoch = epoch
    #             print('New best validation last_bloc_accuracy {}'.format(best_prec1))
    #         else:
    #             print('Current best validation last_bloc_accuracy {}'.format(best_prec1))
                    
    #         save_checkpoint({
    #             'epoch': epoch,
    #             'arch': args.arch,
    #             'state_dict': model.state_dict(),
    #             'best_prec1': best_prec1,
    #             'optimizer': optimizer.state_dict(),
    #         }, args, is_best, model_filename, scores, is_probe=False)

    #     print('Best val_prec1: {:.4f} at epoch {}'.format(best_prec1, best_epoch))
    #     print('Total training time: {}'.format(time.time() - initial_time))

    #     # NEW:
    #     # Collect intermediate predictions after main model training
    #     intermed_data, final_data = collect_validation_predictions(model, val_loader)
    #     print("New validationdataset has been collected")

    #     # Split data into train and validation sets
    #     train_size = int(0.9 * len(final_data))  # 90-10 split
    #     indices = torch.randperm(len(final_data)) # Shuffle indices

    #     train_indices = indices[:train_size]
    #     val_indices = indices[train_size:]

    #     # Split intermediate data
    #     train_intermediate = [data[train_indices] for data in intermed_data]
    #     val_intermediate = [data[val_indices] for data in intermed_data]
    #     train_final = final_data[train_indices]
    #     val_final = final_data[val_indices]
    #     print(f"Split sizes - Train: {len(train_final)}, Val: {len(val_final)}")

    #     # Train and validate probes
    #     temperature = 10.0
    #     best_probe_acc = 0.0
    #     num_probe_epochs = 300

    #     probe_scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_acc\tval_acc']
        
    #     reinitialize_probes(model)
                    
    #     probe_optimizer = torch.optim.SGD([param for probe in model.module.probes for param in probe.parameters()],
    #                        lr=0.001,  # Fixed smaller learning rate
    #                        momentum=args.momentum,
    #                        weight_decay=args.weight_decay)

    #     for epoch in range(num_probe_epochs):
    #         last_checkpoint_num += 1 
    #         model_filename = 'checkpoint_%03d.pth.tar' % last_checkpoint_num

    #         train_loss, train_acc = train_probes(train_intermediate, train_final, model, criterion, probe_optimizer, epoch, temperature=temperature)
    #         val_loss, val_acc = validate_probes(val_intermediate, val_final, model, criterion, temperature=temperature)
    #         print(f"Probe Training Epoch: {epoch}\tTrain Loss: {train_loss:.4f}\tTrain Acc: {train_acc:.4f}\tVal Loss: {val_loss:.4f}\tVal Acc: {val_acc:.4f}")
    #         # Update probe scores
    #         probe_scores.append(('{}\t{:.3f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}').format(
    #             last_checkpoint_num, args.lr, train_loss, val_loss, train_acc, val_acc))
            
    #         # Check if this is the best probe accuracy
    #         is_best = val_acc > best_probe_acc
    #         if is_best:
    #             best_probe_acc = val_acc
    #             print(f'New best probe validation accuracy: {best_probe_acc:.4f}')
            
    #         # Save probe checkpoint
    #         save_checkpoint({
    #             'epoch': last_checkpoint_num,
    #             'arch': args.arch,
    #             'state_dict': model.state_dict(),
    #             'best_probe_acc': best_probe_acc,
    #             'optimizer': optimizer.state_dict(),
    #             'probe_scores': probe_scores
    #         }, args, is_best, model_filename, probe_scores, is_probe=True)

    if not args.compute_only_laplace:
        initial_time = time.time()
        # Track last saved checkpoint number
        last_checkpoint_num = args.epochs - 1  # This will be the last classifier checkpoint number
        
        # 1. First complete model training as before
        for epoch in range(args.start_epoch, args.epochs):

            model_filename = 'checkpoint_%03d.pth.tar' % epoch
            # Train and validate main model 
            train_loss, train_prec1, train_prec5, lr = train(train_loader, model, criterion, optimizer, epoch)
            val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion)
            # Update scores
            scores.append(('{}\t{:.3f}' + '\t{:.4f}' * 6).format(epoch, lr, train_loss, val_loss, train_prec1, val_prec1, train_prec5, val_prec5))

            is_best = val_prec1 > best_prec1
            if is_best:
                best_prec1 = val_prec1
                best_epoch = epoch
                print('New best validation last_bloc_accuracy {}'.format(best_prec1))
            else:
                print('Current best validation last_bloc_accuracy {}'.format(best_prec1))

            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, args, is_best, model_filename, scores, is_probe=False)
        
        print('Best val_prec1: {:.4f} at epoch {}'.format(best_prec1, best_epoch))
        print('Total training time: {}'.format(time.time() - initial_time))

        print("Main model training completed. Starting probe training on validation data...")
        
        # 2. Collect validation predictions after model is fully trained
        val_intermediate_data, val_predictions = collect_validation_predictions(model, val_loader)
        print("Validation predictions collected")

        # 3.Split data into train and validation sets
        train_size = int(0.9 * len(val_predictions))  # 90-10 split
        indices = torch.randperm(len(val_predictions)) # Shuffle indices

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Split intermediate data
        train_intermediate = [data[train_indices] for data in val_intermediate_data]
        val_intermediate = [data[val_indices] for data in val_intermediate_data]
        train_final = val_predictions[train_indices]
        val_final = val_predictions[val_indices]
        print(f"Split sizes - Train: {len(train_final)}, Val: {len(val_final)}")


        # 4. Train probes using validation data
        temperature = 10.0  # Can be tuned
        best_probe_acc = 0.0
        num_probe_epochs = 300

        probe_scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_acc\tval_acc']

        reinitialize_probes(model)

        probe_optimizer = torch.optim.SGD(
            [param for probe in model.module.probes for param in probe.parameters()],
            lr=0.001,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

        for epoch in range(num_probe_epochs):  # You might want to add probe_epochs to args
            last_checkpoint_num += 1 
            model_filename = 'checkpoint_%03d.pth.tar' % last_checkpoint_num
            
            train_loss, train_acc = train_probes_on_val(train_intermediate, train_final, model, criterion, probe_optimizer, epoch, temperature=temperature)
            val_loss, val_acc = validate_probes(val_intermediate, val_final, model, criterion, temperature=temperature)
            print(f"Probe Training Epoch: {epoch}\tTrain Loss: {train_loss:.4f}\tTrain Acc: {train_acc:.4f}\tVal Loss: {val_loss:.4f}\tVal Acc: {val_acc:.4f}")
            # Update probe scores
            probe_scores.append(('{}\t{:.3f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}').format(
                last_checkpoint_num, args.lr, train_loss, val_loss, train_acc, val_acc))
            
            # Save checkpoint if best accuracy
            is_best = val_acc > best_probe_acc
            if is_best:
                best_probe_acc = val_acc
                print(f'New best probe validation accuracy: {best_probe_acc:.4f}')
            
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_probe_acc': best_probe_acc,
                'optimizer': probe_optimizer.state_dict(),
                'probe_scores': probe_scores
            }, args, is_best, model_filename, probe_scores, is_probe=True)

        
    # Load the best model
    model_dir = os.path.join(args.save, 'save_models')
    best_filename = os.path.join(model_dir, 'model_best_acc.pth.tar')

    state_dict = torch.load(best_filename)['state_dict']
    model.load_state_dict(state_dict)

    ### Test the final model
    print('********** Final prediction results with the best model **********')
    validate(test_loader, model, criterion)
    
    ### Test the final model + laplace
    print('********** Precalculate Laplace approximation **********')
    start_time = time.time()
    compute_laplace_efficient(args, model, train_loader)
    print('Laplace computation time: {}'.format(time.time() - start_time))
    return   

def reinitialize_probes(model):
    """Reinitialize probes using MSDNet's existing architecture"""
    new_probes = nn.ModuleList()
    
    # Iterate through existing probes and create new ones with same architecture
    for i, old_probe in enumerate(model.module.probes):
        # Get the input channels from the first conv layer of the existing probe
        first_conv = old_probe.m[0].net[0]  # Access the first conv layer
        in_channels = first_conv.in_channels
        
        if model.module.args.data.startswith('cifar100'):
            new_probe = model.module._build_probe_cifar(in_channels, 100)
        elif model.module.args.data.startswith('cifar10'):
            new_probe = model.module._build_probe_cifar(in_channels, 10)
            
        # Initialize weights
        if hasattr(new_probe, '__iter__'):
            for _m in new_probe:
                model.module._init_weights(_m)
        else:
            model.module._init_weights(new_probe)
            
        new_probes.append(new_probe)
    
    # Replace old probes with new ones
    model.module.probes = new_probes.to(next(model.parameters()).device)

def compute_laplace_efficient(args, model, dset_loader):
    # compute the laplace approximations
    M_W, U, V = get_hessian_efficient(model, dset_loader)
    print(f'Saving the hessians...')
    M_W, U, V = [M_W[i].detach().cpu().numpy() for i in range(len(M_W))], \
                        [U[i].detach().cpu().numpy() for i in range(len(U))], \
                        [V[i].detach().cpu().numpy() for i in range(len(V))]
    np.save(os.path.join(args.save, "effL_llla.npy"), [M_W, U, V])
    

# def collect_intermediate_predictions(model, data_loader):
#     """
#     Collects predictions from each intermediate layer and the final layer to construct
#     the computational uncertainty dataset.

#     Returns data in a format compatible with the existing classifier training structure.
#     """
#     model.eval()
#     intermediate_data = [[] for _ in range(len(model.module.blocks))]  # Initialize directly
#     final_data = []

#     with torch.no_grad():
#         for i, (input, target) in enumerate(data_loader):
#             input = input.cuda()

#             # Initialize lists if it's the first batch
#             if i == 0:
#                 print("\nDebug: Feature collection")
#                 print(f"Input shape: {input.shape}")

#             # Process through blocks and collect features
#             x = input
#             for j in range(len(model.module.blocks)):
#                 x = model.module.blocks[j](x)
#                 intermediate_data[j].append(x[-1])  # Store the feature map that would go to classifier
                
#                 # Debug feature shapes on first batch
#                 if i == 0:
#                     print(f"Block {j} feature shape: {x[-1].shape}")
                
#             # Store final predictions (for training targets)
#             output = model.module.classifier[-1](x)  # Use last classifier's output
#             final_data.append(output)
        
#             if i % 100 == 0:
#                 print(f'Collected data from {i} batches')

#     # Concatenate intermediate predictions from all batches
#     final_data = torch.cat(final_data, dim=0)
#     intermediate_data = [torch.cat(block_data, dim=0) for block_data in intermediate_data]
#     # SAME AS:
#     # for j in range(len(probe_outputs)):
#     #     intermediate_data[j] = torch.cat(intermediate_data[j], dim=0)
    
#     print("\nFinal collected shapes:")
#     for i, data in enumerate(intermediate_data):
#         print(f"Block {i} features shape: {data.shape}")
#     print(f"Final data shape: {final_data.shape}")

#     return intermediate_data, final_data


# (pretty much same as above but cleaner)
def collect_validation_predictions(model, val_loader):
    """
    Collects intermediate features and final predictions from validation data
    """
    model.eval()
    intermediate_data = [[] for _ in range(len(model.module.blocks))]
    final_predictions = []  # Store model's predictions, not true labels

    with torch.no_grad():
        for i, (input, _) in enumerate(val_loader):  # Note: we ignore true labels
            input = input.cuda()
            
            # Process through blocks and collect features
            x = input
            for j in range(len(model.module.blocks)):
                x = model.module.blocks[j](x)
                intermediate_data[j].append(x[-1])
                
            # Store final predictions
            output = model.module.classifier[-1](x)  # Use last classifier's output
            final_predictions.append(output)  # Store raw logits
            
            if i % 100 == 0:
                print(f'Collected validation data from {i} batches')

    # Concatenate all batches
    final_predictions = torch.cat(final_predictions, dim=0)
    intermediate_data = [torch.cat(block_data, dim=0) for block_data in intermediate_data]
    
    return intermediate_data, final_predictions

def train(train_loader, model, criterion, optimizer, epoch):
    global args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to train mode
    model.train()
    end = time.time()
    running_lr = None

    # Add debug prints for model structure
    print("\nClassifier Training Debug Info:")
    print(f"Number of classifiers: {len(model.module.classifier)}")
    
    for i, (input, target) in enumerate(train_loader):
        # print(f"\nInput batch shape for batch {i}: {input.shape}")
        if i == 0:  # Only print for first batch

            # Forward pass to get intermediate shapes
            with torch.no_grad():
                output, intermediate = model(input.cuda(), collect_intermediate=True)
                
                # NEW: Debug intermediate classifier inputs
                print("\n Classifier input shapes:")
                x = input.cuda()
                for j in range(len(model.module.blocks)):
                    x = model.module.blocks[j](x)
                    classifier_input = x[-1]  # The classifier takes the last tensor from each block
                    print(f"Block {j} classifier input shape: {classifier_input.shape}")
                
                # Print shapes of classifier inputs/outputs
                print("\nClassifier output shapes:")
                if not isinstance(output, list):
                    output = [output]
                for j, out in enumerate(output):
                    print(f"Block {j} classifier output shape: {out.shape}")

        total_block_counts = torch.zeros(args.nBlocks)
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)

        if running_lr is None:
            running_lr = lr

        data_time.update(time.time() - end)

        target = target.cuda(device=None)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        loss = 0.0
        # output = model(input_var)
        output, probe_outputs = model(input_var, collect_intermediate=True)  # Changed: Explicitly request intermediate outputs
        
        if not isinstance(output, list):
            output = [output]
           
        for j in range(len(output)):
            loss += criterion(output[j], target_var)

        losses.update(loss.item(), input.size(0))

        for j in range(len(output)):
            prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
            top1[j].update(prec1.item(), input.size(0))
            top5[j].update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.val:.2f}\t'
                  'Acc@1 {top1.val:.1f}\t'
                  'Acc@5 {top5.val:.1f}'.format(
                    epoch, i + 1, len(train_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1[-1], top5=top5[-1]))

    return losses.avg, top1[-1].avg, top5[-1].avg, running_lr

def distillation_loss(student_output, teacher_output, temperature=1.0):
    """Compute the knowledge-distillation (KD) loss."""
    # 1. Soften the teacher's predictions, using probability distributions instead of one-hot labels
    soft_targets = torch.nn.functional.softmax(teacher_output / temperature, dim=1)
    # 2. Soften the student's predictions (in log space)
    student_log_softmax = torch.nn.functional.log_softmax(student_output / temperature, dim=1)
    # 3. Compute KL divergence between soft predictions
    # -(soft_targets * student_log_softmax) is the cross-entropy 
    loss = -(soft_targets * student_log_softmax).sum(dim=1).mean()
    # 4. Scale the loss by temperature squared
    return loss * (temperature ** 2)

# def train_probes(intermediate_data, final_data, model, criterion, optimizer, epoch, temperature):
#     """
#     Train probes for one epoch
#     """
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1, top5 = [], []
#     for i in range(len(model.module.probes)):
#         top1.append(AverageMeter())
#         top5.append(AverageMeter())

#     # switch to train mode
#     for probe in model.module.probes:
#         probe.train()
    
#     end = time.time()
#     running_lr = None
#     batch_size = 64
#     num_samples = final_data.size(0)
#     num_batches = (num_samples + batch_size - 1) // batch_size

#     if epoch == 0:  # Only print debug info in first epoch
#         print("\nProbe Training Debug Info:")
#         print(f"Number of probes: {len(model.module.probes)}")
#         for i, data in enumerate(intermediate_data):
#             print(f"Shape of intermediate_data[{i}]: {data.shape}")
#         print(f"Shape of final_data: {final_data.shape}")


#     for batch_idx, start_idx in enumerate(range(0, num_samples, batch_size)):
#         end_idx = min(start_idx + batch_size, num_samples)
#         data_time.update(time.time() - end)

#         # Adjust learning rate
#         lr = adjust_learning_rate(optimizer, epoch, args, 
#                                 batch=batch_idx,
#                                 nBatch=num_batches, 
#                                 method=args.lr_type)
#         if running_lr is None:
#             running_lr = lr

#         # Get mini-batch of data
#         batch_intermed = [[intermed[start_idx:end_idx].cuda()] for intermed in intermediate_data]
#         # Only take the corresponding samples from final_data
#         batch_final = final_data[start_idx:end_idx].cuda()
        
#         # Debug first batch of first epoch
#         if epoch == 0 and start_idx == 0:
#             print("\nFirst Batch Debug Info:")
#             print("\nBatch shapes:")
#             for i, data in enumerate(batch_intermed):
#                 print(f"Block {i} intermediate batch shape: {data[0].shape}")
#             print(f"Final batch shape: {batch_final.shape}")
            
#             print("\nProbe processing shapes:")
#             for i, (probe, intermed_out) in enumerate(zip(model.module.probes, batch_intermed)):
#                 print(f"\nBlock {i} Probe:")
#                 print(f"  Input shape: {intermed_out[0].shape}")
#                 with torch.no_grad():
#                     probe_out = probe(intermed_out)
#                     print(f"  Output shape: {probe_out.shape}")

#         # Convert final predictions to class indices for accuracy computation
#         _, batch_final_indices = batch_final.max(1)
        
#         # Forward pass through probes
#         total_loss = 0.0
#         num_probes = len(model.module.probes)
#         temp = temperature
#         for i, (probe, intermed_out) in enumerate(zip(model.module.probes, batch_intermed)):
#             probe_out = probe(intermed_out)
#             # total_loss += criterion(probe_out, batch_final)
#             # ALT: Add distillation loss
#             total_loss += distillation_loss(probe_out, batch_final, temperature=temp)
            
#             # Calculate accuracy
#             prec1, prec5 = accuracy(probe_out.data, batch_final_indices, topk=(1, 5))
#             top1[i].update(prec1.item(), batch_size)
#             top5[i].update(prec5.item(), batch_size)

#         losses.update(total_loss.item(), batch_size)

#         # Compute gradient and do SGD step
#         optimizer.zero_grad()
#         total_loss.backward()
#         # Add gradient clipping
#         # torch.nn.utils.clip_grad_norm_([p for probe in model.module.probes for p in probe.parameters()], max_norm=1.0)
#         optimizer.step()

#         # Measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         # Print statistics
#         if batch_idx % args.print_freq == 0:
#             print('Training: [{0}/{1}]\t'
#                   'Time {batch_time.avg:.3f}\t'
#                   'Data {data_time.avg:.3f}\t'
#                   'Loss {loss.val:.4f}\t'
#                   'Avg Acc@1 {top1:.3f}\t'
#                   'Avg Acc@5 {top5:.3f}'.format(
#                     batch_idx + 1, num_batches,
#                     batch_time=batch_time,
#                     data_time=data_time,
#                     loss=losses,
#                     top1=sum(m.avg for m in top1)/len(top1),
#                     top5=sum(m.avg for m in top5)/len(top5),
#                     lr=running_lr))

#     # Print epoch summary
#     epoch_acc = sum(m.avg for m in top1)/len(top1)
#     print(f'\nProbe Training Epoch: {epoch}\t'
#             f'Loss: {losses.avg:.4f}\t'
#             f'Acc: {epoch_acc:.4f}')
    
#     return losses.avg, epoch_acc

def train_probes_on_val(intermediate_data, final_predictions, model, criterion, optimizer, epoch, temperature):
    """
    Train probes using validation data, with model predictions as targets
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(len(model.module.probes)):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to train mode for probes
    for probe in model.module.probes:
        probe.train()
    
    batch_size = 64
    num_samples = final_predictions.size(0)
    num_batches = (num_samples + batch_size - 1) // batch_size

    if epoch == 0:
        print("\nProbe Training on Validation Data Debug Info:")
        print(f"Number of probes: {len(model.module.probes)}")
        for i, data in enumerate(intermediate_data):
            print(f"Shape of intermediate_data[{i}]: {data.shape}")
        print(f"Shape of final_predictions: {final_predictions.shape}")

    # Initialize end time before the loop
    end = time.time()

    for batch_idx, start_idx in enumerate(range(0, num_samples, batch_size)):
        end_idx = min(start_idx + batch_size, num_samples)
        data_time.update(time.time() - end)

        # Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args, 
                                batch=batch_idx,
                                nBatch=num_batches, 
                                method=args.lr_type)

        # Get mini-batch of validation data
        batch_intermed = [[intermed[start_idx:end_idx].cuda()] for intermed in intermediate_data]
        batch_predictions = final_predictions[start_idx:end_idx].cuda()
        
        # Get class indices for accuracy computation
        _, batch_pred_indices = batch_predictions.max(1)
        
        # Forward pass through probes
        total_loss = 0.0
        for i, (probe, intermed_out) in enumerate(zip(model.module.probes, batch_intermed)):
            probe_out = probe(intermed_out)
            # total_loss += criterion(probe_out, batch_final)
            # ALT: Add distillation loss
            # Use distillation loss to match model's predictions
            total_loss += distillation_loss(probe_out, batch_predictions, temperature=temperature)
            
            # Calculate accuracy (compared to model's predictions)
            prec1, prec5 = accuracy(probe_out.data, batch_pred_indices, topk=(1, 5))
            top1[i].update(prec1.item(), batch_size)
            top5[i].update(prec5.item(), batch_size)

        losses.update(total_loss.item(), batch_size)

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        epoch_acc = sum(m.avg for m in top1)/len(top1)

        if batch_idx % args.print_freq == 0:
            print(f'Validation Probe Training: [{batch_idx}/{num_batches}]\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Avg Acc@1 {epoch_acc:.4f}')

    return losses.avg, epoch_acc

def validate_probes(val_intermediate_data, val_final_data, model, criterion, temperature):
    """Validate probes for one epoch"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(len(model.module.probes)):
        top1.append(AverageMeter())
        top5.append(AverageMeter())
    
    # switch to eval mode
    for probe in model.module.probes:
        probe.eval()
    
    batch_size = 64
    num_samples = val_final_data.size(0)
    end = time.time()
    
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            
            # Get mini-batch
            batch_intermed = [[intermed[start_idx:end_idx].cuda()] for intermed in val_intermediate_data]
            batch_final = val_final_data[start_idx:end_idx].cuda()
            
            # Get class indices for accuracy computation
            _, batch_final_indices = batch_final.max(1)
            
            # Forward pass through probes
            total_loss = 0
            temp = temperature
            for i, (probe, intermed_out) in enumerate(zip(model.module.probes, batch_intermed)):
                probe_out = probe(intermed_out)
                total_loss += distillation_loss(probe_out, batch_final, temperature=temp)
                
                # Calculate accuracy
                prec1, prec5 = accuracy(probe_out.data, batch_final_indices, topk=(1, 5))
                top1[i].update(prec1.item(), batch_size)
                top5[i].update(prec5.item(), batch_size)
            
            losses.update(total_loss.item(), batch_size)
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
    epoch_acc = sum(m.avg for m in top1)/len(top1)
    return losses.avg, epoch_acc

def validate(val_loader, model, criterion):

    flops = torch.load(os.path.join(args.save, 'flops.pth'))
    flop_weights = torch.Tensor(flops)/flops[-1]
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5= [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())
    
    n = len(val_loader.sampler)
    confs = torch.zeros(args.nBlocks,n)
    corrs = torch.zeros(args.nBlocks,n)

    '''Addition for uncertainty and exit point computation'''
    # # Matrix to store uncertainty scores for each layer and sample
    # uncertainties = torch.zeros(args.nBlocks, n)
    # # Arrays to store which exit was used and if prediction was correct
    # exit_points = torch.zeros(n, dtype=torch.long)
    # correct_predictions = torch.zeros(n, dtype=torch.bool)

    model.eval()
    end = time.time()
    sample_ind = 0

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            batch_size = input.shape[0]
            target = target.cuda(device=None)
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            # output = model(input_var)
            output, _ = model(input_var, collect_intermediate=True)  # Changed: Explicitly request intermediate outputs
            '''Addition for uncertainty and exit point computation'''
            # # NEW: ALSO COMPUTE CONFIDENCE SCORES
            # outputs, confidences = model.compute_confidence_scores(input_var)
            
            if not isinstance(output, list):
                output = [output]

            '''Addition for uncertainty and exit point computation'''
            # # Store uncertainties (1 - confidence) for each layer
            # for j in range(args.nBlocks):
            #     uncertainties[j, sample_ind:sample_ind+batch_size] = 1 - confidences[j]
            
            loss = 0.0
            for j in range(len(output)):
                loss += criterion(output[j], target_var)

            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(prec1.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))

                '''Addition for uncertainty and exit point computation'''
                # # Store correctness for each sample at this exit
                # for sample in range(batch_size):
                #     pred = output[j][sample].argmax().item()
                #     if pred == target[sample].item():
                #         correct_predictions[sample_ind + sample] = True
                #         if exit_points[sample_ind + sample] == 0:  # If exit point not yet set
                #             exit_points[sample_ind + sample] = j


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            sample_ind += batch_size

            if i % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Acc@1 {top1.val:.4f}\t'
                      'Acc@5 {top5.val:.4f}'.format(
                        i + 1, len(val_loader),
                        batch_time=batch_time, data_time=data_time,
                        loss=losses, top1=top1[-1], top5=top5[-1]))
    
    for j in range(args.nBlocks):
        print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[j], top5=top5[j]))

    '''Addition for uncertainty and exit point computation'''
    # # Save results
    # results_dir = os.path.join(args.save, 'test_results')
    # os.makedirs(results_dir, exist_ok=True)
    # # Save matrices
    # np.save(os.path.join(results_dir, 'uncertainties.npy'), uncertainties.cpu().numpy())
    # np.save(os.path.join(results_dir, 'exit_points.npy'), exit_points.cpu().numpy())
    # np.save(os.path.join(results_dir, 'correct_predictions.npy'), correct_predictions.cpu().numpy())
    # # Save human-readable summary
    # with open(os.path.join(results_dir, 'test_results.txt'), 'w') as f:
    #     f.write('Sample\tExit_Point\tCorrect\tUncertainties\n')
    #     for i in range(n):
    #         uncertainties_str = '\t'.join([f'{u:.4f}' for u in uncertainties[:, i]])
    #         f.write(f'{i}\t{exit_points[i]}\t{int(correct_predictions[i])}\t{uncertainties_str}\n')

    return losses.avg, top1[-1].avg, top5[-1].avg
    
def save_checkpoint(state, args, is_best, filename, result, is_probe=False):
    print(args)
    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = os.path.join(model_dir, filename)

    # Use different best model filename for probes
    if is_probe:
        best_filename = os.path.join(model_dir, 'model_best_probe_acc.pth.tar')
        result_filename = os.path.join(args.save, 'probe_scores.tsv')
    else:
        best_filename = os.path.join(model_dir, 'model_best_acc.pth.tar')
        result_filename = os.path.join(args.save, 'scores.tsv')

    os.makedirs(args.save, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("=> saving checkpoint '{}'".format(model_filename))
    torch.save(state, model_filename)

    with open(result_filename, 'w') as f:
        print('\n'.join(result), file=f)

    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)

    if is_best:
        shutil.copyfile(model_filename, best_filename)

    print("=> saved checkpoint '{}'".format(model_filename))
    return

def load_checkpoint(args):
    model_dir = os.path.join(args.save, 'save_models')
    if args.compute_only_laplace:
        model_filename = os.path.join(model_dir, str(args.resume))
    else:
        latest_filename = os.path.join(model_dir, 'latest.txt')
        if os.path.exists(latest_filename):
            with open(latest_filename, 'r') as fin:
                model_filename = fin.readlines()[0].strip()
        else:
            return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    return state
    
def remove_checkpoints(args, model_filename):
    print("=> removing checkpoints")
    for epoch in range(args.epochs-1):
        filename = 'checkpoint_%03d.pth.tar' % epoch
        model_dir = os.path.join(args.save, 'save_models')
        model_filename = os.path.join(model_dir, filename)
        os.remove(model_filename)
    print("=> checkpoints removed")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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

def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='multistep'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data.startswith('cifar'):
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate ** 2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        elif args.data.startswith('caltech'):
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate ** 2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
    
def Entropy(x):
    # Calculates the mean entropy for a batch of output logits
    epsilon = 1e-5
    p = nn.functional.softmax(x, dim=1)
    Ex = torch.mean(-1*torch.sum(p*torch.log(p), dim=1))
    return Ex

if __name__ == '__main__':
    main()
