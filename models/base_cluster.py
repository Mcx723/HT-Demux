import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict

EPS = 1e-12

class BaseCluster(nn.Module):
    """Abstract base for mixture distributions."""
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

    def init_parameters(self, X: torch.Tensor, counts: Optional[torch.Tensor] = None):
        raise NotImplementedError

    def compute_probability(self, X: torch.Tensor, counts: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

class GMMCluster(BaseCluster):
    """Gaussian distribution for CLR-normalized data."""
    def __init__(self, num_features: int):
        super().__init__(num_features)
        self.mu_sig = nn.Parameter(torch.ones(num_features))
        self.mu_bg = nn.Parameter(torch.zeros(num_features))
        # 使用 log_sigma 确保标准差永远为正
        self.log_sigma_sig = nn.Parameter(torch.zeros(num_features))
        self.log_sigma_bg = nn.Parameter(torch.zeros(num_features))

    def compute_probability(self, X: torch.Tensor, counts: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        def log_gaussian(x, mu, log_sigma):
            # 对齐标准高斯公式：-0.5 * log(2*pi*sigma^2) - (x-mu)^2 / (2*sigma^2)
            sigma = torch.exp(log_sigma)
            var = sigma ** 2 + EPS
            return -0.5 * ((x - mu)**2 / var) - log_sigma - 0.5 * torch.log(torch.tensor(2 * torch.pi))

        ll_sig = log_gaussian(X, self.mu_sig.unsqueeze(0), self.log_sigma_sig.unsqueeze(0))
        ll_bg = log_gaussian(X, self.mu_bg.unsqueeze(0), self.log_sigma_bg.unsqueeze(0))
        return {'sig': ll_sig, 'bg': ll_bg}

    def init_parameters(self, X: torch.Tensor, counts: Optional[torch.Tensor] = None):
        """严格对齐高标准初始化逻辑：使用更具代表性的分位数划分"""
        with torch.no_grad():
            N, H = X.shape
            for h in range(H):
                col = X[:, h]
                # 模拟数据中信号通常位于 95% 分位以上
                q_high = torch.quantile(col, 0.95)
                q_low = torch.quantile(col, 0.50)
                
                high_mask = col > q_high
                low_mask = col <= q_low
                
                # 初始化均值
                m_sig = col[high_mask].mean() if high_mask.any() else col.max()
                m_bg = col[low_mask].mean()
                
                self.mu_sig[h] = m_sig
                self.mu_bg[h] = m_bg
                
                # 初始化标准差：使用各自组内的 std，并设置合理的最小值防止塌陷
                fallback_sd = col.std() if len(col) > 1 else col.new_tensor(0.1)
                s_sig = col[high_mask].std() if high_mask.sum() > 1 else fallback_sd
                s_bg = col[low_mask].std() if low_mask.sum() > 1 else fallback_sd
                
                self.log_sigma_sig[h] = torch.log(torch.clamp(s_sig, min=0.1))
                self.log_sigma_bg[h] = torch.log(torch.clamp(s_bg, min=0.1))

class NBCluster(BaseCluster):
    """Negative Binomial distribution for raw counts."""
    def __init__(self, num_features: int, signal_shape_min: float = 1.0,
                 init_tail_fraction: Optional[float] = None):
        super().__init__(num_features)
        # A positive-signal component must not exploit the zero-count spike of
        # NB(r < 1) to absorb background from rare tags. Background stays free.
        if not np.isfinite(signal_shape_min) or signal_shape_min <= 0:
            raise ValueError('signal_shape_min must be finite and positive')
        if init_tail_fraction is not None and (
                not np.isfinite(init_tail_fraction) or not 0 < init_tail_fraction <= 0.25):
            raise ValueError('init_tail_fraction must be None or in (0, 0.25]')
        self.signal_shape_min = float(signal_shape_min)
        self.init_tail_fraction = init_tail_fraction
        self.mu_sig = nn.Parameter(torch.ones(num_features))
        self.mu_bg = nn.Parameter(torch.ones(num_features) * 0.1)
        self.log_r_sig = nn.Parameter(torch.ones(num_features))
        self.log_r_bg = nn.Parameter(torch.ones(num_features))

    def compute_probability(self, X: torch.Tensor, counts: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        r_s, r_b = torch.exp(self.log_r_sig), torch.exp(self.log_r_bg)
        m_s, m_b = torch.clamp(self.mu_sig, min=EPS), torch.clamp(self.mu_bg, min=EPS)
        
        def log_nb(k, mu, r):
            kd = k.double()
            rd = r.unsqueeze(0).double()
            mud = mu.unsqueeze(0).double()
            
            ll = torch.lgamma(kd + rd) - torch.lgamma(kd + 1.0) - torch.lgamma(rd) + \
                 rd * (torch.log(rd + EPS) - torch.log(rd + mud + EPS)) + \
                 kd * (torch.log(mud + EPS) - torch.log(rd + mud + EPS))
            return ll.float()

        return {'sig': log_nb(counts, m_s, r_s), 'bg': log_nb(counts, m_b, r_b)}

    def init_parameters(self, X: torch.Tensor, counts: Optional[torch.Tensor] = None):
        with torch.no_grad():
            if counts is None: return
            N, H = counts.shape
            for h in range(H):
                col = counts[:, h].float()
                # Seeds only: no cell receives a fixed label here. A fixed top
                # decile is mostly background in highly multiplexed experiments.
                tail = self.init_tail_fraction
                if tail is None:
                    tail = min(0.25, max(1.0 / H, 2.0 / max(N, 1)))
                qv = torch.quantile(col, 1.0 - tail)
                low_mask = col <= torch.quantile(col, 0.5)
                high_mask = col > qv
                
                m_bg = col[low_mask].mean()
                m_sig = col[high_mask].mean() if high_mask.any() else col.max()
                self.mu_bg[h] = torch.clamp(m_bg, min=1e-3)
                self.mu_sig[h] = torch.clamp(m_sig, min=m_bg * 2.0)
                
                def get_r(vals, mu):
                    var = vals.var() if len(vals) > 1 else mu.new_tensor(0.0)
                    r = (mu**2) / torch.clamp(var - mu, min=1.0)
                    return torch.log(torch.clamp(r, min=0.1, max=100.0))

                self.log_r_bg[h] = get_r(col[low_mask], self.mu_bg[h])
                self.log_r_sig[h] = torch.clamp(get_r(col[high_mask], self.mu_sig[h]),
                                                min=np.log(self.signal_shape_min))


# Discrete count emission: kept here to preserve the existing package layout.
"""Discrete lognormal count components with exact bin probabilities and weighted MLE."""
import numpy as np
import torch
from torch import nn
from scipy.special import log_ndtr
from scipy.optimize import minimize


def _normal_bin_logprob(lower,upper,mu,sd):
    a=(lower-mu)/sd;b=(upper-mu)/sd
    right=a>0
    lo=np.where(right,-b,a);hi=np.where(right,-a,b)
    lhi=log_ndtr(hi);llo=log_ndtr(lo)
    return lhi+np.log(-np.expm1(llo-lhi)),a,b


class DiscreteLogNormalCluster(BaseCluster):
    """K=max(0,round(exp(Z)-1)), Z~Normal(mu,sigma), separately per tag/state."""
    def __init__(self,h):
        super().__init__(h)
        self.mu_sig=nn.Parameter(torch.ones(h)*6)
        self.mu_bg=nn.Parameter(torch.ones(h))
        self.log_sigma_sig=nn.Parameter(torch.zeros(h))
        self.log_sigma_bg=nn.Parameter(torch.zeros(h))

    def compute_probability(self,X,counts=None):
        k=counts.double()
        lower=torch.log(k+.5);lower=torch.where(k==0,torch.full_like(lower,-torch.inf),lower)
        upper=torch.log(k+1.5)
        def mass(mu,logsigma):
            sd=logsigma.double().exp();a=(lower-mu.double())/sd;b=(upper-mu.double())/sd
            right=a>0;lo=torch.where(right,-b,a);hi=torch.where(right,-a,b)
            lhi=torch.special.log_ndtr(hi);llo=torch.special.log_ndtr(lo)
            return lhi+torch.log(-torch.expm1(llo-lhi))
        return {'sig':mass(self.mu_sig,self.log_sigma_sig),'bg':mass(self.mu_bg,self.log_sigma_bg)}

    @torch.no_grad()
    def init_parameters(self,X,counts=None):
        n,h=counts.shape;y=torch.log1p(counts)
        for j in range(h):
            col=counts[:,j];tail=min(.25,max(1/h,2/max(n,1)))
            sig=col>torch.quantile(col,1-tail);bg=col<=torch.quantile(col,.9)
            v=y[sig,j] if sig.any() else y[:,j]
            self.mu_sig[j]=v.mean();self.mu_bg[j]=y[bg,j].mean()
            self.log_sigma_sig[j]=v.std(unbiased=False).clamp_min(.5).log()
            self.log_sigma_bg[j]=y[bg,j].std(unbiased=False).clamp_min(.5).log()

    @torch.no_grad()
    def weighted_mstep(self,stats,signal):
        sig=signal.cpu().numpy();gain=0.
        for kind,table in [('sig',sig),('bg',np.maximum(stats.total_np-sig,0))]:
            means=getattr(self,'mu_'+kind).double().cpu().numpy().copy()
            logs=getattr(self,'log_sigma_'+kind).double().cpu().numpy().copy()
            for h,w in enumerate(table):
                active=w>0;mass=w.sum()
                if mass<1e-6:continue
                k=stats.values_np[active];w=w[active]/mass
                lo=np.log(k+.5);lo[k==0]=-np.inf;hi=np.log(k+1.5)
                old=np.array([means[h],logs[h]])
                def q(v):
                    sd=np.exp(v[1]);lp,a,b=_normal_bin_logprob(lo,hi,v[0],sd)
                    pa=np.exp(-.5*a*a-.5*np.log(2*np.pi)-lp)
                    pb=np.exp(-.5*b*b-.5*np.log(2*np.pi)-lp)
                    aa=np.where(np.isfinite(a),a,0.)
                    grad_mu=(pa-pb)/sd
                    grad_sigma=aa*pa-b*pb
                    return -float(w@lp),-np.array([w@grad_mu,w@grad_sigma])
                fit=minimize(q,old,jac=True,method='L-BFGS-B',bounds=[(-12,32),(np.log(.1),np.log(5.))],options={'maxiter':80,'ftol':1e-12,'gtol':1e-8})
                before=q(old)[0];after=q(fit.x)[0]
                if np.isfinite(after) and after<=before:
                    means[h]=fit.x[0];logs[h]=fit.x[1];gain+=mass*(before-after)
            getattr(self,'mu_'+kind).copy_(torch.as_tensor(means,device=self.mu_sig.device,dtype=self.mu_sig.dtype))
            getattr(self,'log_sigma_'+kind).copy_(torch.as_tensor(logs,device=self.mu_sig.device,dtype=self.mu_sig.dtype))
        return gain/len(stats.counts)
