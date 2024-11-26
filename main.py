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
    probe_optimizer = torch.optim.SGD([param for probe in model.module.probes for param in probe.parameters()],
                           lr=0.001,  # Fixed smaller learning rate
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

        if args.evalmode == 'anytime':
            validate(test_loader, model, criterion)
        else:
            dynamic_evaluate(model, test_loader, val_loader, args)
        return

    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_prec1'
              '\tval_prec1\ttrain_prec5\tval_prec5']

    if not args.compute_only_laplace:
        initial_time = time.time()

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
            }, args, is_best, model_filename, scores)

        print('Best val_prec1: {:.4f} at epoch {}'.format(best_prec1, best_epoch))
        print('Total training time: {}'.format(time.time() - initial_time))

        # NEW:
        # Collect intermediate predictions after main model training
        intermed_data, final_data = collect_intermediate_predictions(model, train_loader)
        print("New dataset has been collected")

        # Split data into train and validation sets
        train_size = int(0.9 * len(final_data))  # 90-10 split
        indices = torch.randperm(len(final_data)) # Shuffle indices

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Split intermediate data
        train_intermediate = [data[train_indices] for data in intermed_data]
        val_intermediate = [data[val_indices] for data in intermed_data]
        train_final = final_data[train_indices]
        val_final = final_data[val_indices]
        print(f"Split sizes - Train: {len(train_final)}, Val: {len(val_final)}")

        # Train and validate probes
        best_val_acc = 0
        num_probe_epochs = 10
        for epoch in range(num_probe_epochs):
            train_loss, train_acc = train_probes(train_intermediate, train_final, model, criterion, probe_optimizer, epoch)
            val_loss, val_acc = validate_probes(val_intermediate, val_final, model, criterion)
            
            print(f"Probe Training Epoch: {epoch}\tTrain Loss: {train_loss:.4f}\tTrain Acc: {train_acc:.4f}\tVal Loss: {val_loss:.4f}\tVal Acc: {val_acc:.4f}")
            
            # Track best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"New best validation accuracy: {best_val_acc:.4f}")

        
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

def compute_laplace_efficient(args, model, dset_loader):
    # compute the laplace approximations
    M_W, U, V = get_hessian_efficient(model, dset_loader)
    print(f'Saving the hessians...')
    M_W, U, V = [M_W[i].detach().cpu().numpy() for i in range(len(M_W))], \
                        [U[i].detach().cpu().numpy() for i in range(len(U))], \
                        [V[i].detach().cpu().numpy() for i in range(len(V))]
    np.save(os.path.join(args.save, "effL_llla.npy"), [M_W, U, V])


def collect_intermediate_predictions(model, data_loader):
    """
    Collects predictions from each intermediate layer and the final layer to construct
    the computational uncertainty dataset.

    Returns data in a format compatible with the existing classifier training structure.
    """
    model.eval()
    intermediate_data = [[] for _ in range(len(model.module.blocks))]  # Initialize directly
    final_data = []

    with torch.no_grad():
        for i, (input, target) in enumerate(data_loader):
            input = input.cuda()

            # Initialize lists if it's the first batch
            if i == 0:
                print("\nDebug: Feature collection")
                print(f"Input shape: {input.shape}")

            # Process through blocks and collect features
            x = input
            for j in range(len(model.module.blocks)):
                x = model.module.blocks[j](x)
                intermediate_data[j].append(x[-1])  # Store the feature map that would go to classifier
                
                # Debug feature shapes on first batch
                if i == 0:
                    print(f"Block {j} feature shape: {x[-1].shape}")
                
            # Store final predictions (for training targets)
            output = model.module.classifier[-1](x)  # Use last classifier's output
            final_data.append(output)
        
            if i % 100 == 0:
                print(f'Collected data from {i} batches')

    # Concatenate intermediate predictions from all batches
    final_data = torch.cat(final_data, dim=0)
    intermediate_data = [torch.cat(block_data, dim=0) for block_data in intermediate_data]
    # SAME AS:
    # for j in range(len(probe_outputs)):
    #     intermediate_data[j] = torch.cat(intermediate_data[j], dim=0)
    
    print("\nFinal collected shapes:")
    for i, data in enumerate(intermediate_data):
        print(f"Block {i} features shape: {data.shape}")
    print(f"Final data shape: {final_data.shape}")

    return intermediate_data, final_data



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
    soft_targets = torch.nn.functional.softmax(teacher_output / temperature, dim=1)
    student_log_softmax = torch.nn.functional.log_softmax(student_output / temperature, dim=1)
    loss = -(soft_targets * student_log_softmax).sum(dim=1).mean()
    return loss * (temperature ** 2)

def train_probes(intermediate_data, final_data, model, criterion, optimizer, epoch):
    """Train probes for one epoch"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(len(model.module.probes)):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to train mode
    for probe in model.module.probes:
        probe.train()
    
    end = time.time()
    running_lr = None
    batch_size = 64
    num_samples = final_data.size(0)
    num_batches = (num_samples + batch_size - 1) // batch_size

    if epoch == 0:  # Only print debug info in first epoch
        print("\nProbe Training Debug Info:")
        print(f"Number of probes: {len(model.module.probes)}")
        for i, data in enumerate(intermediate_data):
            print(f"Shape of intermediate_data[{i}]: {data.shape}")
        print(f"Shape of final_data: {final_data.shape}")


    for batch_idx, start_idx in enumerate(range(0, num_samples, batch_size)):
        end_idx = min(start_idx + batch_size, num_samples)
        data_time.update(time.time() - end)

        # Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args, 
                                batch=batch_idx,
                                nBatch=num_batches, 
                                method=args.lr_type)
        if running_lr is None:
            running_lr = lr

        # Get mini-batch of data
        batch_intermed = [[intermed[start_idx:end_idx].cuda()] for intermed in intermediate_data]
        # Only take the corresponding samples from final_data
        batch_final = final_data[start_idx:end_idx].cuda()
        
        # Debug first batch of first epoch
        if epoch == 0 and start_idx == 0:
            print("\nFirst Batch Debug Info:")
            print("\nBatch shapes:")
            for i, data in enumerate(batch_intermed):
                print(f"Block {i} intermediate batch shape: {data[0].shape}")
            print(f"Final batch shape: {batch_final.shape}")
            
            print("\nProbe processing shapes:")
            for i, (probe, intermed_out) in enumerate(zip(model.module.probes, batch_intermed)):
                print(f"\nBlock {i} Probe:")
                print(f"  Input shape: {intermed_out[0].shape}")
                with torch.no_grad():
                    probe_out = probe(intermed_out)
                    print(f"  Output shape: {probe_out.shape}")

        # Convert final predictions to class indices for accuracy computation
        _, batch_final_indices = batch_final.max(1)
        
        # Forward pass through probes
        total_loss = 0.0
        num_probes = len(model.module.probes)
        for i, (probe, intermed_out) in enumerate(zip(model.module.probes, batch_intermed)):
            probe_out = probe(intermed_out)
            # total_loss += criterion(probe_out, batch_final)
            # ALT: Add distillation loss
            total_loss += distillation_loss(probe_out, batch_final, temperature=2.0)
            
            # Calculate accuracy
            prec1, prec5 = accuracy(probe_out.data, batch_final_indices, topk=(1, 5))
            top1[i].update(prec1.item(), batch_size)
            top5[i].update(prec5.item(), batch_size)

        losses.update(total_loss.item(), batch_size)

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_([p for probe in model.module.probes for p in probe.parameters()], max_norm=1.0)
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print statistics
        if batch_idx % args.print_freq == 0:
            print('Training: [{0}/{1}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.val:.4f}\t'
                  'Avg Acc@1 {top1:.3f}\t'
                  'Avg Acc@5 {top5:.3f}'.format(
                    batch_idx + 1, num_batches,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=sum(m.avg for m in top1)/len(top1),
                    top5=sum(m.avg for m in top5)/len(top5),
                    lr=running_lr))

    # Print epoch summary
    epoch_acc = sum(m.avg for m in top1)/len(top1)
    print(f'\nProbe Training Epoch: {epoch}\t'
            f'Loss: {losses.avg:.4f}\t'
            f'Acc: {epoch_acc:.4f}')
    
    return losses.avg, epoch_acc


def validate_probes(val_intermediate_data, val_final_data, model, criterion):
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
            for i, (probe, intermed_out) in enumerate(zip(model.module.probes, batch_intermed)):
                probe_out = probe(intermed_out)
                total_loss += distillation_loss(probe_out, batch_final, temperature=2.0)
                
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
            output, probe_outputs = model(input_var, collect_intermediate=True)  # Changed: Explicitly request intermediate outputs
            if not isinstance(output, list):
                output = [output]

            loss = 0.0
            for j in range(len(output)):
                loss += criterion(output[j], target_var)

            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(prec1.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))

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

    return losses.avg, top1[-1].avg, top5[-1].avg
    
def save_checkpoint(state, args, is_best, filename, result):
    print(args)
    result_filename = os.path.join(args.save, 'scores.tsv')
    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = os.path.join(model_dir, filename)
    best_filename = os.path.join(model_dir, 'model_best_acc.pth.tar')

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
