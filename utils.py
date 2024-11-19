import numpy as np

def truncation(L,tau,mrs,r,volvol,v_bar,rho,sigma):
    '''
    Finds the integration range for the COS method
    L : truncation parameter
    '''
    print(f"L: {type(L)}")
    print(f"TAU: {type(tau)}")
    print(f"MRS: {type(mrs)}")
    print(f"VolVol: {type(volvol)}")
    print(f"V/Bar: {type(v_bar)}")
    print(f"Rho: {type(rho)}")
    print(f"Sigma: {type(sigma)}")
    c1 = r* tau + (1 - np.exp(-mrs * tau)) * (v_bar - sigma)/(2 * mrs) - v_bar * tau / 2 # this is the first order cumulant of the characterisctic function of the log-asset price

    c2 = 1/(8 * np.power(mrs,3)) * (volvol * tau * mrs * np.exp(-mrs * tau) \
        * (sigma - v_bar) * (8 * mrs * rho - 4 * volvol) \
        + mrs * rho * volvol * (1 - np.exp(-mrs * tau)) * (16 * v_bar - 8 * sigma) \
        + 2 * v_bar * mrs * tau * (-4 * mrs * rho * volvol + np.power(volvol,2) + 4 * np.power(mrs,2)) \
        + np.power(volvol,2) * ((v_bar - 2 * sigma) * np.exp(-2 * mrs * tau) \
        + v_bar * (6 * np.exp(-mrs * tau) - 7) + 2 * sigma) \
        + 8 * np.power(mrs,2) * (sigma - v_bar) * (1 - np.exp(-mrs * tau))) # this is the second order cumulant of the characterisctic function of the log-asset price

    a = c1 - L * np.sqrt(np.abs(c2))
    b = c1 + L * np.sqrt(np.abs(c2))
    return a, b
def cosSerExp(a,b,c,d,k):
    '''
    The cosine series coefficients of g(y)=exp(y) on [c,d] included in [a,b]
    k : positive integer

    '''
    bma = b-a
    uu  = k * np.pi/bma
    chi =  (1/(1 + np.power(uu,2)))*(np.cos(uu*(d-a))*np.exp(d) - np.cos(uu*(c-a))*np.exp(c) + uu*np.sin(uu*(d-a))*np.exp(d) - uu*np.sin(uu*(c-a))*np.exp(c))

    return chi
def cosSer1(a,b,c,d,k):
    '''
    The cosine series coefficients of g(y)=1 on [c,d] included in [a,b]
    k : positive integer
  
    '''
    bma    = b-a
    uu     = k * np.pi/bma
    uu[0]  = 1
    psi    = (1/uu)*(np.sin(uu*(d-a)) - np.sin(uu*(c-a)))
    psi[0] = d-c
    return psi

if __name__ == "__main__":
    # Example usage with numerical inputs
    L = 3.0
    tau = 1.5
    mrs = 0.05
    r = 0.03
    volvol = 0.1
    v_bar = 0.2
    rho = 0.5
    sigma = 0.01

    a, b = truncation(L, tau, mrs, r, volvol, v_bar, rho, sigma)
    print(a, b)
