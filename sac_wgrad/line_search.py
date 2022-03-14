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
# from modules.utils import _safe_norm

def _safe_norm(v):
    if not torch.isfinite(v).all():
        return np.inf
    return torch.norm(v)

def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    ite = 0
    phi_a0 = phi(alpha0)    # First do an update with step size 1
    mask = phi_a0 > phi0 + c1*alpha0*derphi0
    if torch.sum(mask)==0:
        return alpha0, phi_a0, ite

    alpha1 = mask*alpha0/2.0 + (~mask)*alpha0
    alpha2 = alpha1
    phi_a1 = phi(alpha1)

    while torch.min(alpha1) > amin:       # we are assuming alpha>0 is a descent direction
        phi_a2 = phi(alpha2)
        ite += 1
        mask = phi_a2 > phi0 + c1*alpha2*derphi0
        if torch.sum(mask)==0:
            return alpha2, phi_a2, ite

        alpha2 = mask*alpha1/2.0 + (~mask)*alpha1

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2
    mask = alpha1 < amin
    alpha1 = (~mask)*alpha1

    # Failed to find a suitable step length
    return alpha1, phi_a1, ite

def line_search_lagr(update, xzs0, mu, rho, g0, lagrfunc, g, nstep=0, on=True, default=True, vec=True):
    """
    `update` is the proposed direction of update.

    Code adapted from scipy.
    """
    tmp_s = [0]
    s_norm = torch.norm(xzs0) / torch.norm(update)

    if on:
        with torch.enable_grad():
            xzs_est = Variable(xzs0, requires_grad=True).to(xzs0.device)
            _, lagr = lagrfunc(xzs_est, mu, rho, vec=vec)
        tmp_phi = [lagr]  
        tmp_g0 = [g0]

    def phi(s, store=True):
        # if s == tmp_s[0]:
        #     return tmp_phi[0]    # If the step size is so small... just return something
        if vec:
            xzs_est = xzs0 + s.unsqueeze(-1) * update
        else:
            xzs_est = xzs0 + s * update
        with torch.enable_grad():
            xzs_est_ = Variable(xzs_est, requires_grad=True).to(xzs_est.device)
            _, loss = lagrfunc(xzs_est_, mu, rho, vec=vec)
        phi_new = loss
        if store:
            tmp_s[0] = s
            tmp_phi[0] = phi_new
        return phi_new
    if on:
        derphi = torch.sum(update*g0, dim=1)
        derphi = derphi if vec else derphi.mean()
        alpha = torch.ones(tmp_phi[0].shape[0], dtype=tmp_phi[0].dtype, device=tmp_phi[0].device) if vec else 1.
        derphi0 = -tmp_g0[0]
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], derphi, amin=1e-4, alpha0=alpha)
    else:
        s = 1.0
        ite = 0
    if s is None:
        if default is True:
            s = 1.0
            ite = 0
        else:
            return torch.zeros_like(xzs_est), 0, 0

    if vec:
        xzs_est = xzs0 + s.unsqueeze(-1) * update
        return xzs_est - xzs0, ite, torch.mean(s)
    else:
        xzs0 + s * update
        return xzs_est - xzs0, ite, s



# def line_search_lagr(update, xzs0, mu, rho, g0, lagrfunc, g, mu_lr=0.1, nstep=0, on=True, default=True):
#     """
#     `update` is the proposed direction of update.

#     Code adapted from scipy.
#     """
#     tmp_s = [0]
#     s_norm = torch.norm(xzs0) / torch.norm(update)

#     if on:
#         with torch.enable_grad():
#             xzs_est = Variable(xzs0, requires_grad=True).to(xzs0.device)
#             _, lagr = lagrfunc(xzs_est, mu, rho)
#         tmp_phi = [lagr]  
#         tmp_g0 = [g0]

#     def phi(s, store=True):
#         if s == tmp_s[0]:
#             return tmp_phi[0]    # If the step size is so small... just return something
#         xzs_est = xzs0 + s * update
#         with torch.enable_grad():
#             xzs_est_ = Variable(xzs_est, requires_grad=True).to(xzs_est.device)
#             _, loss = lagrfunc(xzs_est_, mu, rho)
#         phi_new = loss
#         if store:
#             tmp_s[0] = s
#             tmp_phi[0] = phi_new
#         return phi_new
#     if on:
#         derphi = torch.sum(update*g0, dim=1).mean()
#         derphi0 = -tmp_g0[0]
#         s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], derphi, amin=1e-4)
#     else:
#         s = 1.0
#         ite = 0
#     if s is None:
#         if default is True:
#             s = 1.0
#             ite = 0
#         else:
#             return torch.zeros_like(xzs_est), 0, 0

#     xzs_est = xzs0 + s * update
#     return xzs_est - xzs0, ite, s


# def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
#     ite = 0
#     phi_a0 = phi(alpha0)    # First do an update with step size 1
#     if phi_a0 <= phi0 + c1*alpha0*derphi0:
#         return alpha0, phi_a0, ite

#     alpha1 = alpha0/2.0
#     alpha2 = alpha1
#     phi_a1 = phi(alpha1)

#     while alpha1 > amin:       # we are assuming alpha>0 is a descent direction
#         phi_a2 = phi(alpha2)
#         ite += 1

#         if (phi_a2 <= phi0 + c1*alpha2*derphi0):
#             return alpha2, phi_a2, ite

#         alpha2 = alpha1 / 2.0

#         alpha0 = alpha1
#         alpha1 = alpha2
#         phi_a0 = phi_a1
#         phi_a1 = phi_a2

#     # Failed to find a suitable step length
#     return None, phi_a1, ite


def line_search(update, x0, z0, mu, rho, g0, diffunc, g, mu_lr = 1, nstep=0, on=True, default=True):
    """
    `update` is the proposed direction of update.

    Code adapted from scipy.
    """
    x_size = x0.shape[1]
    z_size = z0.shape[1]
    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [torch.norm(g0)**2]
    xz0 = torch.cat([x0, z0], dim = 1)
    s_norm = torch.norm(xz0) / torch.norm(update)
    with torch.enable_grad():
        xz_est = Variable(xz0, requires_grad=True).to(xz0.device)
        g0_new = g(xz_est[:,x_size:], xz_est[:,:x_size], mu, rho)
        # grad = torch.autograd.grad(g0_new.norm()**2, xz_est)[0]

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]    # If the step size is so small... just return something
        xz_est = xz0 + s * update
        with torch.enable_grad():
            xz_est_ = Variable(xz_est, requires_grad=True).to(xz_est.device)
            g0_new = g(xz_est_[:,x_size:], xz_est_[:,:x_size], mu, rho)
        phi_new = _safe_norm(g0_new)**2
        if store:
            tmp_s[0] = s
            tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new
    if on:
        # derphi = torch.sum(update*grad, dim=1).mean()
        derphi0 = -tmp_phi[0]
        # print("derphi : ", derphi.item())
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], derphi0, amin=1e-4)
    else:
        s = 1.0
        ite = 0
    if s is None:
        if default is True:
            s = 1.0
            ite = 0
        else:
            diff, _ = diffunc(xz_est[:,x_size:], xz_est[:,:x_size], mu, rho)
            mu_old = mu
            mu = mu + mu_lr*rho*diff
            with torch.enable_grad():
                xz_est_ = Variable(xz_est, requires_grad=True).to(xz_est.device)
                g0_new = g(xz_est_[:,x_size:], xz_est_[:,:x_size], mu_old, rho)
            return torch.zeros_like(xz0), g0_new, torch.zeros_like(g0), 0, mu, 0.

    xz_est = xz0 + s * update
    diff, _ = diffunc(xz_est[:,x_size:], xz_est[:,:x_size], mu, rho)
    mu_old = mu
    mu = mu + mu_lr*rho*diff
    with torch.enable_grad():
        xz_est_ = Variable(xz_est, requires_grad=True).to(xz_est.device)
        # g0_new = g(xz_est_[:,x_size:], xz_est_[:,:x_size], mu, rho)
        g0_old = g(xz_est_[:,x_size:], xz_est_[:,:x_size], mu_old, rho)

    return xz_est - xz0, g0_old, g0_old - g0, ite, mu, s


# def line_search_lagr(update, x0, z0, mu, rho, g0, lagrfunc, g, mu_lr = 1, nstep=0, on=True, default=True):
#     """
#     `update` is the proposed direction of update.

#     Code adapted from scipy.
#     """
#     x_size = x0.shape[1]
#     z_size = z0.shape[1]
#     tmp_s = [0]
#     xz0 = torch.cat([x0, z0], dim = 1)
#     s_norm = torch.norm(xz0) / torch.norm(update)
#     with torch.enable_grad():
#         xz_est = Variable(xz0, requires_grad=True).to(xz0.device)
#         _, lagr = lagrfunc(xz_est[:,x_size:], xz_est[:,:x_size], mu, rho)
#         grad = torch.autograd.grad(lagr, xz_est)[0]
#     tmp_phi = [lagr]
#     tmp_grad = [grad.norm()**2]  

#     def phi(s, store=True):
#         if s == tmp_s[0]:
#             return tmp_phi[0]    # If the step size is so small... just return something
#         xz_est = xz0 + s * update
#         with torch.enable_grad():
#             xz_est_ = Variable(xz_est, requires_grad=True).to(xz_est.device)
#             _, loss = lagrfunc(xz_est_[:,x_size:], xz_est_[:,:x_size], mu, rho)
#             grad = torch.autograd.grad(loss, xz_est_)[0]
#         phi_new = loss
#         if store:
#             tmp_s[0] = s
#             tmp_grad[0] = grad.norm()**2
#             tmp_phi[0] = phi_new
#         return phi_new
#     if on:
#         derphi = torch.sum(update*grad, dim=1).mean()
#         derphi0 = -tmp_grad[0]
#         # print("grad_norm : ", grad.norm().item(), "update_norm : ", update.norm().item())
#         # print("derphi : ", derphi.item())
#         s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], derphi, amin=1e-5)
#         # print(lagr.item(), tmp_phi[0].item())
#         # print("s : ", s)
#     else:
#         s = 1.0
#         ite = 0
#     if s is None:
#         if default is True:
#             s = 1.0
#             ite = 0
#         else:
#             diff, _ = lagrfunc(xz_est[:,x_size:], xz_est[:,:x_size], mu, rho)
#             mu_old = mu
#             mu = mu + mu_lr*rho*diff
#             with torch.enable_grad():
#                 xz_est_ = Variable(xz_est, requires_grad=True).to(xz_est.device)
#                 g0_new = g(xz_est_[:,x_size:], xz_est_[:,:x_size], mu_old, rho)
#             return torch.zeros_like(xz0), g0_new, torch.zeros_like(g0), 0, mu, 0.
#     xz_est = xz0 + s * update
#     diff, _ = lagrfunc(xz_est[:,x_size:], xz_est[:,:x_size], mu, rho)
#     mu_old = mu
#     mu = mu + mu_lr*rho*diff
#     with torch.enable_grad():
#         xz_est_ = Variable(xz_est, requires_grad=True).to(xz_est.device)
#         # g0_new = g(xz_est_[:,x_size:], xz_est_[:,:x_size], mu, rho)
#         g0_old = g(xz_est_[:,x_size:], xz_est_[:,:x_size], mu_old, rho)

#     return xz_est - xz0, g0_old, g0_old - g0, ite, mu, s

def line_search_lagr_slack(update, x0, z0, mu, rho, slack, g0, lagrfunc, g, mu_lr=0.1, nstep=0, on=True, default=True):
    """
    `update` is the proposed direction of update.

    Code adapted from scipy.
    """
    x_size = x0.shape[1]
    z_size = z0.shape[1]
    xz_size = x_size + z_size
    tmp_s = [0]
    if slack is None:
        xzs0 = torch.cat([x0, z0], dim = 1)
    else:
        xzs0 = torch.cat([x0, z0, slack], dim = 1)
    s_norm = torch.norm(xzs0) / torch.norm(update)

    if on:
        with torch.enable_grad():
            xzs_est = Variable(xzs0, requires_grad=True).to(xzs0.device)
            slack_ = None if slack is None else xzs_est[:, xz_size:]
            _, lagr = lagrfunc(xzs_est[:,x_size:xz_size], xzs_est[:,:x_size], mu, rho, slack_)
        tmp_phi = [lagr]  
        tmp_g0 = [g0]

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]    # If the step size is so small... just return something
        xzs_est = xzs0 + s * update
        with torch.enable_grad():
            xzs_est_ = Variable(xzs_est, requires_grad=True).to(xzs_est.device)
            slack_ = None if slack is None else xzs_est[:, xz_size:]
            _, loss = lagrfunc(xzs_est_[:,x_size:xz_size], xzs_est_[:,:x_size], mu, rho, slack_)
        phi_new = loss
        if store:
            tmp_s[0] = s
            tmp_phi[0] = phi_new
        return phi_new
    if on:
        derphi = torch.sum(update*g0, dim=1).mean()
        derphi0 = -tmp_g0[0]
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], derphi, amin=1e-4)
    else:
        s = 1.0
        ite = 0
    if s is None:
        if default is True:
            s = 1.0
            ite = 0
        else:
            # with torch.enable_grad():
            #     xzs_est_ = Variable(xzs0, requires_grad=True).to(xzs0.device)
            #     slack_ = None if slack is None else xzs_est_[:, xz_size:]
            #     g0_new = g(xzs_est_[:,x_size:xz_size], xzs_est_[:,:x_size], mu, rho, slack_)
            return torch.zeros_like(xzs0), 0, 0

    xzs_est = xzs0 + s * update
    return xzs_est - xzs0, ite, s


def line_search_lagr_slack_sampled(update, x0, z0, z_sampled, eps_sampled, mu, rho, slack, g0, lagrfunc, g, args, mu_lr=0.1, nstep=0, on=True, default=True):
    """
    `update` is the proposed direction of update.

    Code adapted from scipy.
    """

    x_size = x0.shape[1]
    z_size = z0.shape[1]
    xz_size = x_size + z_size
    # ipdb.set_trace()
    # updatez = update[:, x_size:xz_size].reshape(-1, args.num_samples, z_size)
    # updatez = torch.mean(updatez, dim=1, keepdim=True).repeat(1, args.num_samples, 1).reshape(-1, z_size)
    # update[:, x_size:xz_size] = updatez
    tmp_s = [0]
    if slack is None:
        xzs0 = torch.cat([x0, z0], dim = 1)
    else:
        xzs0 = torch.cat([x0, z0, slack], dim = 1)
    s_norm = torch.norm(xzs0) / torch.norm(update)

    with torch.enable_grad():
        xzs_est = Variable(xzs0, requires_grad=True).to(xzs0.device)
        slack_ = None if slack is None else xzs_est[:, xz_size:]
        _, lagr = lagrfunc(z0, x0, mu, rho, slack = slack_)
    tmp_phi = [lagr]  
    tmp_g0 = [g0]

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]    # If the step size is so small... just return something
        xzs_est = xzs0 + s * update
        with torch.enable_grad():
            xzs_est_ = Variable(xzs_est, requires_grad=True).to(xzs_est.device)
            z_est_ = xzs_est_[:,x_size:xz_size]
            # z_sampled_ = z_est_[:, :z_est_.shape[1]//2] + eps_sampled * torch.exp(0.5 * z_est_[:, z_est_.shape[1]//2:])
            slack_ = None if slack is None else xzs_est[:, xz_size:]
            _, loss = lagrfunc(z_est_, xzs_est_[:,:x_size], mu, rho, slack=slack_)
        phi_new = loss
        if store:
            tmp_s[0] = s
            tmp_phi[0] = phi_new
        return phi_new
    if on:
        derphi = torch.sum(update*g0, dim=1).mean()
        derphi0 = -tmp_g0[0]
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], derphi, amin=1e-4)
    else:
        s = 1.0
        ite = 0
    if s is None:
        if default is True:
            s = 1.0
            ite = 0
        else:
            with torch.enable_grad():
                xzs_est_ = Variable(xzs0, requires_grad=True).to(xzs0.device)
                slack_ = None if slack is None else xzs_est_[:, xz_size:]
                eps_sampled_new = eps_sampled# torch.randn((z0.shape[0], z0.shape[1]//2)).to(x0.device)
                z_est_ = xzs_est_[:,x_size:xz_size]
                z_sampled_new = z_sampled#z_est_[:, :z_est_.shape[1]//2] + eps_sampled_new * torch.exp(0.5 * z_est_[:, z_est_.shape[1]//2:])
                g0_new = g(z_est_, xzs_est_[:,:x_size], mu, rho, slack = slack_)
            return torch.zeros_like(xzs0), g0_new, torch.zeros_like(g0), 0, 0, z_sampled_new, eps_sampled_new
    xzs_est = xzs0 + s * update
    with torch.enable_grad():
        xzs_est_ = Variable(xzs_est, requires_grad=True).to(xzs_est.device)
        eps_sampled_new = eps_sampled# torch.randn((z0.shape[0], z0.shape[1]//2)).to(x0.device)
        z_est_ = xzs_est_[:,x_size:xz_size]
        z_sampled_new = z_sampled# z_est_[:, :z_est_.shape[1]//2] + eps_sampled_new * torch.exp(0.5 * z_est_[:, z_est_.shape[1]//2:])
        slack_ = None if slack is None else xzs_est_[:, xz_size:]
        g0_new = g(z_est_, xzs_est_[:,:x_size], mu, rho, slack=slack_)
    return xzs_est - xzs0, g0_new, g0_new - g0, ite, s, z_sampled_new, eps_sampled_new


def line_search_vanilla(update, x0, g0, g, nstep=0, on=True):
    """
    `update` is the propsoed direction of update.

    Code adapted from scipy.
    """
    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [torch.norm(g0)**2]
    s_norm = torch.norm(x0) / torch.norm(update)

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]    # If the step size is so small... just return something
        x_est = x0 + s * update
        g0_new = g(x_est)
        phi_new = _safe_norm(g0_new)**2
        if store:
            tmp_s[0] = s
            tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new
    
    if on:
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=1e-14)
    if (not on) or s is None:
        s = 1.0
        ite = 0
        print(s)

    x_est = x0 + s * update
    if s == tmp_s[0]:
        g0_new = tmp_g0[0]
    else:
        g0_new = g(x_est)
    return x_est - x0, g0_new, g0_new - g0, ite, s