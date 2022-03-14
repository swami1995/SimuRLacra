import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function, Variable
import sys
import os
from scipy.optimize import root
import time
# from termcolor import colored
import ipdb
import numpy as np
from line_search import *
from tensorboardX import SummaryWriter


def newton_al(g, diffunc, z0, x0, sT, args, rand_dirs, ls=True, name="unknown", rho=None, mu = None, nstep = None, writer = None):
    '''
    Input:
        g : function which returns the gradient of the lagrangian
        diffunc : returns the lagrangian and constraint values
        z0 : initial value of z
        x0 : initial value of x
        threshold : maximum number of iterations of optimization allowed
        eps : threshold value for the lagrangian gradient for the exit condition 
        ls : whether to perform line search
        rho, mu, nstep : initial values of corresponding quantities
    '''
    # Initializing Variables and coefficients
    vectorize = True
    verbose = True
    bsz, x_size = x0.size()
    _, z_size = z0.size()
    x_est = x0.detach()   
    z_est = z0.detach()
    xz_est = torch.cat([x_est, z_est], dim=1)
    if mu is None:
        mu = torch.zeros(bsz, x_size, dtype = x0.dtype).to(x0.device)
    # mu_const = torch.zeros(bsz, x_size, dtype = x0.dtype).to(x0.device)
    if rho is None:
        if vectorize:
            rho = torch.ones((bsz,1), dtype=x0.dtype).to(x0.device)*args.rho
        else:
            rho = torch.ones((1,1), dtype=x0.dtype).to(x0.device)*args.rho
    args.rho_max = rho*args.rho_max
    I = torch.eye(x_size+z_size, dtype = x0.dtype).to(x0.device)
    if nstep is None:
        nstep = 0
    tnstep = 0
    nstep_inner = 0

    # Saving stuff
        # rho_str = str(args.rho).replace('.', '')
        # mulr_str = str(args.mu_lr).replace('.', '')
        # lin_diff_str = "_lindiff" if args.lin_diff else ""
        # mu_tol_str = str(args.mu_update_tol).replace('.', '')
        # save_folder =  args.save_folder + "/newton/rhoinit" + rho_str + "_rhoratio" + str(args.rho_ratio) + "_mulr" + mulr_str + "_mutol" + mu_tol_str +  "_rhomax" + str(args.rho_max) + lin_diff_str + "/"
        # save = not args.train and args.verbose and args.save_optimizer
        # args.opt_folder = None
        # if os.path.isdir(save_folder) and save:
        #     ipdb.set_trace()
        # if save:
        #     if writer is None:
        #         writer = SummaryWriter(save_folder) 
        #         os.system('cp modules/optimizers.py ' + save_folder)
        #         os.system('cp modules/deq.py ' + save_folder)
        #         os.system('cp SimpleDEQ.py ' + save_folder)
        #         os.system('cp modules/line_search.py ' + save_folder)
        #         os.system('cp deq_wrappers.py ' + save_folder)
        #         os.system('cp models.py ' + save_folder)
        #         args.opt_folder = save_folder

    
    def newton_step(g, x_est, z_est, mu, rho):
        # Computing the newton update = - J^(-1) gx. using cholesky decomposition for computing the jacobian inverse.
        grads = []
        with torch.enable_grad():
            x_est = Variable(x_est, requires_grad = True).to(x_est.device)
            z_est = Variable(z_est, requires_grad = True).to(x_est.device)
            xz_est = torch.cat([x_est, z_est], dim=1)
            gx, diff, lagr = g(xz_est, mu, rho, sT, x_size, rand_dirs)    
            if torch.isnan(gx).sum():
                ipdb.set_trace()
            for i in range(gx.shape[1]):
                output = torch.zeros(gx.shape[1], dtype = x0.dtype).to(x_est.device)
                output[i] = 1.
                # ipdb.set_trace()
                grad = torch.autograd.grad(gx.sum(0)+z_est.norm()*0., [x_est, z_est], grad_outputs = output, retain_graph=True)
                grad = torch.cat(grad, dim=1)
                if torch.isnan(grad).sum():
                    ipdb.set_trace()
                grads.append(grad)
            
        xz_est_grad = torch.stack(grads, dim=1)  
        # ipdb.set_trace() 
        reg = 1e-5
        while True:   
            try:
                if reg>100000:
                    ipdb.set_trace()
                    break
                u = torch.linalg.cholesky(xz_est_grad + reg*I)
                if torch.isnan(u).sum() == 0:
                    break
                else:
                    reg = reg*10
            except:
                reg = reg*10
        update = -torch.cholesky_solve(gx.unsqueeze(2), u).squeeze()
        # print(torch.diagonal(xz_est_grad, dim1=-2, dim2=-1).abs().mean())
        return update, gx, reg, diff, lagr

    # Computing the update
    update, gx, reg, diff, lagr = newton_step(g, x_est, z_est, mu, rho)
    new_objective = lagr.item()
    init_objective = gx.norm().item()
    grad_objective = init_objective
    prot_break = False
    trace = [init_objective]


    # Initializing more variables that track progress
    protect_thres = 1e4
    lowest = new_objective
    lowest_xzest, lowest_gx, lowest_step = xz_est, gx, nstep
    lowest_diff = diff.norm().item()
    diffnorm = lowest_diff
    # if verbose:
    #     print("step,    gx,      mu,      lagr,     diff_norm,     reg,   stepsize,   x_norm,   z_norm, gradx_norm, gradz_norm")
    #     print(nstep, " {:.6f} {:.6f}    {:.6f}    {:.6f}   {:.2e}   {:.6f}   {:.6f}   {:.6f}   {:.6f}   {:.6f} ".format(gx.norm().item(), mu.norm().item(), lagr.item(), diff.norm().item(), reg, 0., x_est.norm().item(), z_est.norm().item(), gx[:, :x_size].norm().item(), gx[:, x_size:].norm().item()))
        
    while (grad_objective >= 1e-8 or diff.norm().item()>1e-8) and nstep < 60:
        # Performing Line search and the corresponding update, and updating all the tracking variables
        gx_old = gx
        lagrfunc = lambda x, mu, rho, vec: diffunc(x, mu, rho, sT, x_size, rand_dirs, vec)
        delta_xz, ite, s = line_search_lagr(update, xz_est, mu, rho, gx, lagrfunc, g, nstep=nstep, on=ls, default=False)

        x_est += delta_xz[:,:x_size]
        z_est += delta_xz[:,x_size:]
        xz_est = torch.cat([x_est, z_est], dim=1)
        nstep += 1
        nstep_inner += 1
        tnstep += (ite+1)
        with torch.enable_grad():
            xz_est_ = Variable(xz_est, requires_grad=True).to(xz_est.device)
            gx, diff, lagr = g(xz_est_, mu, rho, sT, x_size, rand_dirs)
        delta_gx = gx - gx_old
        new_objective = lagr
        grad_objective = gx.norm().item()
        diffnorm = diff.norm().item()
        trace.append(grad_objective)
        # if verbose:
            # print(nstep, " {:.6f} {:.6f}    {:.6f}     {:.6f}   {:.2e}   {:.6f}   {:.6f}   {:.6f}   {:.6f}   {:.6f} ".format(gx.norm().item(), mu.norm().item(), lagr.item(), diffnorm, reg, s, x_est.norm().item(), z_est.norm().item(), gx[:, :x_size].norm().item(), gx[:, x_size:].norm().item()))
            # print(x_est.abs().max())
        # Save the current values if constraint satisfaction is considerably better or if the objective value decreased without affecting the constraint values much.
        if (new_objective < lowest and diff.norm() < lowest_diff*1.5) or diff.norm() < 0.8*lowest_diff or nstep<10:
            lowest_xzest, lowest_gx = xz_est.clone().detach(), gx.clone().detach()
            lowest = min(new_objective, lowest)
            lowest_step = nstep
            lowest_diff = min(diff.norm().item(), lowest_diff)
            rho_best, mu_best = rho, mu

        if vectorize:
            mu_update_mask = (gx.norm(dim=-1)<args.mu_update_tol).unsqueeze(-1).float()
        else:
            mu_update_mask = (gx.norm()<args.mu_update_tol).unsqueeze(-1).unsqueeze(-1).float()
        mu = mu + rho*diff*mu_update_mask
        new_rho = torch.minimum(rho*args.rho_ratio, args.rho_max)
        rho = rho*(1-mu_update_mask) + new_rho*mu_update_mask
        
        # Computing the update for the next iteration
        update, gx, reg, diff, lagr = newton_step(g, x_est, z_est, mu, rho)
        diffnorm = diff.norm().item()

        # Exit criterion : (1) the KKT conditions are satisfied or (2) there's not been much progress in the last 30 iterations
        if grad_objective < 1e-8 and diffnorm < 1e-8:
            break
        if nstep > 5 and np.max(trace[-5:]) / np.min(trace[-5:]) < 1.005:
            # if there's hardly been any progress in the last 5 steps
            break
        if grad_objective > init_objective * protect_thres or reg > 100:
            prot_break = True
            break


    args.result = [lowest, lowest_diff]
    args.rho_final = rho_best
    args.mu_final = mu_best
    # if save and not args.tune_hp:
        # print("do you want to save or delete this run? Yes: 0, No: 1")
        # input1 = int(input())
        # if input1==0:
        #     os.system("rm -rf " + save_folder)
        #     args.opt_folder = None
    # ipdb.set_trace()
    return {"result": lowest_xzest,
            "nstep": nstep,
            "tnstep": tnstep,
            "lowest_step": lowest_step,
            "diff": torch.norm(lowest_gx).item(),
            "diff_detail": torch.norm(lowest_gx, dim=1),
            "prot_break": prot_break,
            "trace": trace}


def bfgs_al(g, diffunc, z0, x0, s0, args, threshold, eps, ls=True, name="unknown", implicit=True):
    '''
    Input:
        g : function which returns the gradient of the lagrangian
        g_newton : The gradient function for newton if used for finetuning.
        diffunc : returns the lagrangian and constraint values
        z0 : initial value of z
        x0 : initial value of x
        threshold : maximum number of iterations of optimization allowed
        eps : threshold value for the lagrangian gradient for the exit condition 
        ls : whether to perform line search
        implicit : whether to compute the hessian implicitly using history or explicitly store a moving value of the hessian approximation
    '''
    # Saving stuff
    vectorize = False
    start = time.time()
    args.rho = 1
    args.rho_max = 1e6
    args.mu_update_tol = 1e-6
    args.lbfgs_mem = 10

    # Initializing Variables and coefficients
    bsz, x_size = x0.size()
    _, z_size = z0.size()
    xz_size = x_size + z_size
    total_hsize = z_size + x_size
    mu = torch.zeros(bsz, x_size#+z_size
                                , dtype=x0.dtype).to(x0.device)
    # mu_const = torch.zeros(bsz, x_size+z_size, dtype=x0.dtype).to(x0.device)
    if vectorize:
        rho = torch.ones((bsz,1), dtype=x0.dtype).to(x0.device)*args.rho
    else:
        rho = torch.ones((1,1), dtype=x0.dtype).to(x0.device)*args.rho
    args.rho_max = rho*args.rho_max
    I = torch.eye(x_size+z_size, dtype = x0.dtype).to(x0.device)
    x_est = x0 
    z_est = z0
    nstep = 0
    nstep_inner = 0
    tnstep = 0
    num_fails = 0
    L_const = torch.tensor([0.])
    xzs_est = torch.cat([x_est, z_est], dim=1)
    lm = []
    for i in range(0, threshold):
        s = torch.zeros(bsz, total_hsize, dtype=x0.dtype).to(x0.device)
        y = torch.zeros(bsz, total_hsize, dtype=x0.dtype).to(x0.device)
        lm.append(iterationData(s[:,0], s, y, s[:,0]))

    # Computing the first update direction
    with torch.enable_grad():
        xzs_est_ = Variable(xzs_est, requires_grad = True).to(x_est.device)
        gx, diff, lagr = g(xzs_est_, mu, rho, s0, x_size)
    update = -gx      

    # Initializing more variables to keep track of progress
    new_objective = lagr.item()
    init_objective = gx.norm().item()
    prot_break = False
    trace = [init_objective]
    if not implicit:
        C = torch.stack([torch.eye(xzs_est.shape[1])]*bsz, dim=0).to(x0.device)
        I = C.clone()
    
    # To be used in protective breaks
    protect_thres = 1e5
    lowest = new_objective
    lowest_xzest, lowest_gx, lowest_step = xzs_est, gx, nstep
    lowest_diff = diff.norm().item()
    end = 0
    cont = False
    slack_norm = 0.
    lowest_diff_stored = lowest_diff
    lowest_gx_stored = gx.norm().item()
    num_fails_total = 0
    lagrangian = diffunc(xzs_est, mu, rho)[1]

    if True:
        interval = time.time()-start
        print("step,    gx,        lagr,       mu,   errx,   errfx,  constraint,    stepsize, num_fails,   xnorm,    znorm")
        print(nstep, "      {:.6f} {:.10f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}".format(gx[:, :xz_size].norm().item(), lagrangian.item(), mu.norm().item(), lagr.item(), lagr.item(), diff.norm().item(), 0., num_fails_total, xzs_est[:,:x_size].norm().item(), xzs_est[:, x_size:].norm().item(), gx[:, :x_size].norm().item(), gx[:, x_size:].norm().item()))
        # print(xzs_est[0, :20])
    while nstep < threshold: 
        # Performing Line search and the corresponding update, and updating all the tracking parameters
        gx_old = gx
        lagrfunc = lambda x, mu, rho, vec: diffunc(x, mu, rho, s0, x_size, vec)
        delta_xzs, ite, s = line_search_lagr(update, xzs_est, mu, rho, gx, diffunc, g, nstep=nstep, on=ls, default=False)

        xzs_est += delta_xzs
        lagrangian = diffunc(xzs_est, mu, rho)[1]

        with torch.enable_grad():
            xzs_est_ = Variable(xzs_est, requires_grad=True).to(xzs_est.device)
            gx, diff, lagr = g(xzs_est_, mu, rho, s0, x_size)
        delta_gx = gx - gx_old
        Bs = -s*gx_old
        nstep += 1
        tnstep += (ite+1)
        new_objective = lagr.item()
        grad_objective = gx.norm().item()
        trace.append(grad_objective)

        # Save the current values if constraint satisfaction is considerably better or if the objective value decreased without affecting the constraint values much.
        if (new_objective < lowest and diff.norm() < lowest_diff*1.5) or diff.norm() < 0.8*lowest_diff:
            lowest_xzest, lowest_gx = xzs_est.clone().detach(), gx.clone().detach()
            lowest = lowest
            lowest_step = nstep
            lowest_diff = min(diff.norm().item(), lowest_diff)
            lowest_diff_stored = diff.norm().item()
            lowest_gx_stored = gx.norm().item()
            rho_best, mu_best = rho, mu

        # If the gradient of the lagrangian is below a threshold, then update mu, rho and free the buffers.

        # if grad_objective<args.mu_update_tol:# and nstep<300:
        #     mu = mu + rho*diff
        #     rho = min(rho*args.rho_ratio, args.rho_max)
        #     with torch.enable_grad():
        #         xzs_est_ = Variable(xzs_est, requires_grad = True).to(x_est.device)
        #         gx = g(xzs_est_, mu, rho, return_all=False)
        #         if args.generate:
        #             gx[:,x_size:] *= 0         
        #         cont = True
        #         update = -gx
        #         grad_objective = gx.norm().item()


        # if s==0:
        #     cont = True
        #     update = -gx

        if True:
            interval = time.time() - start
            slack_norm = 0.
            print(nstep, "      {:.6f} {:.10f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}".format(gx[:, :xz_size].norm().item(), lagrangian.item(), mu.norm().item(), lagr.item(), lagr.item(), diff.norm().item(), s, num_fails_total,xzs_est[:,:x_size].norm().item(), xzs_est[:, x_size:].norm().item(), gx[:, :x_size].norm().item(), gx[:, x_size:].norm().item()))
            # print(xzs_est[0, :20])
            # print(gx_old[:, x_size:].norm().item(), update[:, x_size:].norm().item())
            # print(xzs_est[:, :x_size].norm().item(), xzs_est[:, x_size:].norm().item(), update.norm().item())

        if cont:# and nstep_inner > 2:
            cont = False
            nstep_inner = 0
            continue

        ## Computing the BFGS update
        # update vectors s and y:
        nstep_inner += 1
        it = lm[nstep_inner-1]
        it.s = delta_xzs

        # Checking positive definiteness and damping
        it.y = delta_gx
        ys = torch.bmm(it.y.view(bsz, 1, -1), it.s.view(bsz, -1, 1)).squeeze(-1)
        sBs = torch.bmm(it.s.view(bsz, 1, -1), Bs.view(bsz, -1, 1)).squeeze(-1)

        if (ys<0.2*sBs).sum() > 0:
            # nstep_inner -= 1
            num_fails_total += (ys<=0).sum()
            num_fails += 1
            damping = True
            if damping:
                theta = torch.ones_like(ys)
                theta[ys<0.2*sBs] = (((1 - 0.2) * sBs)/torch.clamp(sBs - ys, 1e-14, 100)) [ys<0.2*sBs]
                it.y = theta * it.y + (1 - theta) * Bs
                ys = torch.bmm(it.y.view(bsz, 1, -1), it.s.view(bsz, -1, 1)).squeeze(-1)
        if torch.isnan(ys.norm()):
            ipdb.set_trace()

        if vectorize:
            mu_update_mask = (gx.norm(dim=-1)<args.mu_update_tol).unsqueeze(-1).float()
        else:
            mu_update_mask = (gx.norm()<args.mu_update_tol*np.sqrt(bsz)).unsqueeze(-1).unsqueeze(-1).float()
            if mu_update_mask[0,0] == 1:
                # if nstep_inner < 2:
                #     args.mu_update_tol*=0.5
                #     print("changing tolerance", nstep_inner)
                nstep_inner=0
                
        mu = mu + rho*diff*mu_update_mask
        new_rho = torch.minimum(rho*args.rho_ratio, args.rho_max)
        rho = rho*(1-mu_update_mask) + new_rho*mu_update_mask
        with torch.enable_grad():
            xzs_est_ = Variable(xzs_est, requires_grad=True).to(xzs_est.device)
            gx, diff, lagr = g(xzs_est_, mu, rho, s0, x_size)

        # For the limited memory version, uncomment the second line
        # bound = nstep_inner 
        bound = min(args.lbfgs_mem,nstep_inner)

        # Compute scalars ys and yy:
        yy = torch.bmm(it.y.view(bsz, 1, -1), it.y.view(bsz, -1, 1)).squeeze(-1)
        it.ys = ys
        # print(ys.norm().item(), yy.norm().item())
        update = -gx
        j = nstep_inner
        if nstep > 300:
            ipdb.set_trace()
        if implicit:
            for i in range(0, bound):
                # from later to former
                j = j-1
                it = lm[j]
                it.alpha = torch.bmm(it.s.view(bsz, 1, -1), update.view(bsz, -1, 1)).squeeze(-1) / (it.ys + 1e-8)
                update = update + (it.y * (-it.alpha))
            update = update * (ys/(yy + 1e-8))
            # newton_step(g, xzs_est, mu, rho, update)

            for i in range(0, bound):
                it = lm[j]
                beta = torch.bmm(it.y.view(bsz, 1, -1), update.view(bsz, -1, 1)).squeeze(-1)
                beta = beta /(it.ys + 1e-8)
                update = update + (it.s * (it.alpha - beta))
                # from former to later
                j = j+1
        else:
            ys = ys.unsqueeze(-1)
            bias = torch.bmm(it.s.view(bsz, -1, 1), it.s.view(bsz, 1, -1))/(ys + 1e-12)
            lft = I - torch.bmm(it.s.view(bsz, -1, 1), it.y.view(bsz, 1, -1))/(ys + 1e-12)
            rht = I - torch.bmm(it.y.view(bsz, -1, 1), it.s.view(bsz, 1, -1))/(ys + 1e-12)
            C = torch.bmm(torch.bmm(lft, C), rht) + bias
            update = torch.bmm(C, update.view(bsz, -1, 1)).squeeze()
        if nstep > 300:
            ipdb.set_trace()


        # Exit criterion : (1) the KKT conditions are satisfied or (2) there's not been much progress in the last 30 iterations
        if grad_objective < 1e-6 and diff.norm() < 1e-6:
            # ipdb.set_trace()
            break
        if nstep > 50 and np.max(trace[-30:]) / np.min(trace[-30:]) < 1.01:
            # if there's hardly been any progress in the last 30 steps
            break
        if grad_objective > init_objective * protect_thres:
            prot_break = True
            break
        if num_fails_total > 1000:
            break

    #### If we want to finetune with Newton
    args.result = [lowest, lowest_diff]
    args.rho_final = rho
    args.mu_final = mu
    args.diff_final = lowest_diff_stored
    args.gx_final = lowest_gx_stored
    args.n_steps = nstep
    args.num_fails_total = num_fails_total
    ipdb.set_trace()

    return {"result": lowest_xzest,
            "nstep": nstep,
            "tnstep": tnstep,
            "lowest_step": lowest_step,
            "diff": torch.norm(lowest_gx).item(),
            "diff_detail": torch.norm(lowest_gx, dim=1),
            "prot_break": prot_break,
            "trace": trace,
            "eps": eps,
            "threshold": threshold, 
            "slack_norm": slack_norm}
