# A collection of routines for the CORSIKA airshower package
# (C) 2022, 2023 Kael D. HANSON

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# CORSIKA I/O routines

class CORSIKAParticle:
    part_id = { 1: "gamma", 2: "e+", 3: "e-", 5:"mu+", 6: "mu-", 7: "pi0", 
                8: "pi+", 9: "pi-", 13: "n", 14: "p"}
    def __init__(self, part):
        id_had_lev = int(part[0])
        self.particle_id  = id_had_lev // 1000
        self.hadronic_ctr = (id_had_lev // 10) % 100
        self.obs_level    = id_had_lev % 10
        self.p = part[1:4]
        self.x = part[4]
        self.y = part[5]
        self.t = part[6]
        
    def __repr__(self):
        p3 = np.sum(self.p**2)**0.5
        p_id = f'{self.particle_id}'
        if self.particle_id in CORSIKAParticle.part_id:
            p_id = CORSIKAParticle.part_id[self.particle_id]
        return p_id + f' {p3:.2f} GeV/c {self.x:.0f} {self.y:.0f}'
    
class CORSIKAReader:
    def readTape(self, f, maxrec=1000000000):
        for self.frec in range(maxrec):
            self.nbytes = int.from_bytes(f.read(4), 'little')
            if self.nbytes == 0: return
            self.buf = f.read(self.nbytes)
            i = 0
            while i < self.nbytes:
                try:
                    blkhdr = self.buf[i:i+4].decode('ASCII')
                    self.__dict__[blkhdr] = np.frombuffer(self.buf[i+4:i+1092], dtype='float32')
                    self.blockHandler(blkhdr)
                except UnicodeDecodeError:
                    part = np.frombuffer(self.buf[i:i+1092], dtype='float32').reshape(-1,7)
                    for p in part:
                        if p[0] != 0.0: self.particleHandler(CORSIKAParticle(p))
                i += 1092
            end_marker = int.from_bytes(f.read(4), 'little')
            
    def blockHandler(self, blk):
        pass
    
    def particleHandler(self, part):
        pass
    
    @property
    def runNumber(self):
        return int(self.RUNH[0])
    
    @property
    def eventNumber(self):
        return int(self.EVTH[0])
    

def dPdh(h, p, atm):
    return -p * atm.M_m * atm.g / (atm.R * atm.temperature(h))

class Atmosphere:
    T0  = 273.16 + 15.0     # Sea level temperature
    P0  = 101325.0          # Sea level pressure (Pa)
    M_m = 0.028966          # Molar mass of dry air (kg/mol)
    R   = 8.314             # Ideal gas constant
    g   = 9.80665           # Specific accel. of gravity
    
    def __init__(self, h=None, **kwparms):
        if h is None: h = np.arange(0, 50000, 100)
        self.h = h
        self.pressure_solution = None
        self.P0 = Atmosphere.P0
        if 'SLP' in kwparms: self.P0 = kwparms['SLP']
        
    @property
    def pressure(self):
        if self.pressure_solution is None:
            self.pressure_solution = solve_ivp(dPdh, (self.h[0], self.h[-1]), (self.P0,), t_eval=self.h, args=(self,))
        return self.pressure_solution.y[0]
    
    @property
    def overburden(self):
        """
        Mass thickness in g/cm**2
        """
        return self.pressure / self.g / 10
    
    @property
    def density(self):
        p = self.pressure
        t = np.array([self.temperature(h) for h in self.h], 'd')
        return p * self.M_m / (self.R * t)
        
    # default - isothermal atmosphere
    def temperature(self, h):
        return self.T0
    
class ISA(Atmosphere):
    """
    International Standard Atmosphere: non-isothermal atmosphere.
    """
    def __init__(self, h=None):
        super().__init__(h)
        h_s = np.array((0, 11000, 20000, 32000, 47000, 100000), 'd')
        T_s = np.array((288.16, 216.66, 216.66, 228.66, 270.66, 270.66), 'd')
        self.T_int = interp1d(h_s, T_s)
        
    def temperature(self, h):
        return self.T_int(h)
        #if h < 11000: return self.T0 - 0.0065*h
        #if h < 20000: return 216.66
        #if h <= 32000: return 216.66 + (h-20000)*0.001
        #if h <= 47000: return 228.66 + (h-32000)*0.0028
        #return 270.66

class CORSIKAAtmosphere:
    def __init__(self, layers=None, par=None):
        if par is None: par = np.zeros((3, 4), 'd')
        self._par = par
        self._layers = layers
        
    def zoneIndex(self, h):
        """
        Returns vector of indices to map into the 4 zones.
        """
        return np.sum((h >= self._layers[:,np.newaxis]).astype('i'),axis=0)-1
     
    def __call__(self, h, *par):
        """
        Returns the thickness for given height(s), h.
        """
        if len(par) == 0:
            par = self._par.reshape((3,4))
        else:
            par = np.array(par, 'd')
            par.shape = (3,4)
            
        idx = self.zoneIndex(h)
        return par[0,idx] + par[1,idx]*np.exp(-h/par[2,idx])
    
    def temperature(self, h):
        idx = self.zoneIndex(h)
        return Atmosphere.M_m * Atmosphere.g / Atmosphere.R * self._par[2,idx] * \
            (1 + self._par[0,idx]/self._par[1,idx]*np.exp(h/self._par[2,idx]))
    
    @staticmethod
    def fitToAtmosphere(atm):
        self = CORSIKAAtmosphere()
        self._atm = atm
        self._layers = atm.h
        h = atm.h
        a = np.zeros(4, 'd')
        b = np.zeros(4, 'd')
        c = np.zeros(4, 'd')
        Winv = atm.R / (atm.M_m * atm.g)
        for i in range(1, 5):
            dh = h[i] - h[i-1]
            dt = atm.temperature(h[i]) - atm.temperature(h[i-1])
            a_ov_b = dt/dh * Winv
            c[i-1] = atm.temperature(h[i-1]) * Winv / (1 + a_ov_b)
            a[i-1] = (atm.overburden[i] - atm.overburden[i-1]*np.exp(-dh/c[i-1]))/(1 - np.exp(-dh/c[i-1]))
            b[i-1] = (atm.overburden[i-1] - a[i-1])*np.exp(h[i-1]/c[i-1])            
        self._par = np.vstack((a, b, c))
        return self
        
    def improveFit(self, atm):
        fit_par, fit_pcov = curve_fit(self, atm.h, atm.overburden, p0 = self._par.flatten())
        self._par = np.array(fit_par)
        self._par.shape = (3, 4)
        
    @property
    def ATMA(self):
        "Generate CORSIKA ATMA steering card"
        return "ATMA    " + " ".join([f"{x:.6g}" for x in self._par[0]])
    
    @property
    def ATMB(self):
        "Generate CORSIKA ATMB steering card"
        return "ATMB    " + " ".join([f"{x:.6g}" for x in self._par[1]])
    
    @property
    def ATMC(self):
        "Generate CORSIKA ATMC steering card"
        return "ATMC    " + " ".join([f"{100*x:.6g}" for x in self._par[2]])
    
    @property
    def ATMLAY(self):
        "Generate CORSIKA ATMLAY steering card"
        layers = self._layers[1:]
        return "ATMLAY  " + " ".join([f"{100.0*x:.4e}" for x in layers])
    
    