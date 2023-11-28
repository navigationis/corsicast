# A collection of routines for the CORSIKA airshower package
# (C) 2022, 2023 Kael D. HANSON

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import logging

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
    def __init__(self, f, endianness='little'):
        self.f = f
        self.irec = 0
        self.isub = 0
        self.end = endianness

    def __iter__(self):
        return self
    
    def __next__(self):
        while True:
            self.isub %= 21
            if self.isub == 0:
                nb = int.from_bytes(self.f.read(4), self.end)
                logging.debug(f'Record start, bytes = {nb}')
                if nb == 0: raise StopIteration
                if nb != 22932: raise ValueError("Invalid block size")
            
            buf = self.f.read(1092)
            self.isub += 1
            if self.isub == 21: 
                end_marker = int.from_bytes(self.f.read(4), 'little')
                logging.debug(f'Record end, marker = {end_marker}')

            match buf[0:4]:
                case b'RUNH' | b'EVTH' | b'LONG' | b'EVTE' | b'RUNE':
                    blk = buf[:4].decode('ASCII')
                    logging.debug(f'sub block {self.isub} type ' + blk)
                    self.__dict__[blk] = np.frombuffer(buf[4:], dtype=np.float32)
                    if blk == 'EVTH': self.part = []
                    if blk == 'EVTE': return self
                case other:
                    # It's a particle data block
                    part = np.frombuffer(buf, dtype=np.float32).reshape(-1, 7)
                    for p in part:
                        if p[0] != 0.0: self.part.append(CORSIKAParticle(p))
                    logging.debug(f'sub block {self.isub} is DATABLOCK particle {part[0][0]:.0f}')
    
    @property
    def runNumber(self):
        return int(self.RUNH[0])
    
    @property
    def eventNumber(self):
        return int(self.EVTH[0])
    

def dPdh(h, p, atm):
    return -p * atm.M_m * atm.g / (atm.R * atm.temperature(h))

class Atmosphere:
    M_m = 0.028966          # Molar mass of dry air (kg/mol)
    R   = 8.314             # Ideal gas constant
    g   = 9.80665           # Specific accel. of gravity
    
    def __init__(self, h=None, SLP=101325.0, T0=273.16+15):
        if h is None: h = np.arange(0, 100000, 1000)
        self.h = h
        self.pressure_solution = None
        self.P0 = SLP
        self.T0 = T0

    @property
    def h(self):
        return self._h
      
    @h.setter
    def h(self, val):
        self._h = val
        self.pressure_solution = None

    @property
    def pressure(self):
        if self.pressure_solution is None:
            self.pressure_solution = solve_ivp(dPdh, (0, self.h[-1]), (self.P0,), t_eval=self.h, args=(self,))
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
        h_s = np.array((0, 11000, 20000, 32000, 47000, 51000, 71000, 85000, 200000), 'd')
        T_s = np.array((19, -56.5, -56.5, -44.5, -2.5, -2.5, -58.5, -86.28, 1000.0), 'd') + 273.13
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
        return np.sum((h >= self._layers[:,np.newaxis]).astype('i'), axis=0)
     
    def __call__(self, h, *par):
        """
        Returns the thickness for given height(s), h.
        """
        if len(par) > 0:
            par = np.array(par, 'd')
            par.resize((3, 4))
        else:
            par = self._par

        idx = self.zoneIndex(h)
        return par[0,idx] + par[1,idx]*np.exp(-h/par[2,idx])
    
    def temperature(self, h):
        idx = self.zoneIndex(h)
        return Atmosphere.M_m * Atmosphere.g / Atmosphere.R * self._par[2,idx] * \
            (1 + self._par[0,idx]/self._par[1,idx]*np.exp(h/self._par[2,idx]))
    
    @staticmethod
    def fitToAtmosphere(atm):
        """
        Redefine the CORSIKA parameters for the $i$ zones as 
        $$X_i(h) = a_i + b'_i\exp[-(h-h_{i-1})/c_i]$$
        where $b'_i \equiv b_i\exp(h_{i-1}/c_i)$
        In each zone, the parameter $c_i$ is defined in terms of the known starting 
        temperature, $T_{i-1}$:
        $$c_i = \frac{RT_{i-1}}{M_m g}$$
        $b'_i$ can then be found from the above definition, and $a_i$ by forcing 
        the function to fit the endpoints:
        \begin{equation}
        \begin{split}
        b'_i &= X_{i-1} - a_i \\ 
        a_i  &= \frac{X_i - X_{i-1}\exp(-\Delta h_i/c_i)}{1-\exp(- \Delta h_i/c_i)}\\
        \end{split}
        \end{equation}

        Finally, the CORSIKA $b_i$ can then be found from $b'_i$ via the above definition.
        """
        self = CORSIKAAtmosphere()
        self._atm = atm
        self._layers = atm.h
    
        h = atm.h
        h0 = 0.0
        X0 = 0.1 * atm.P0 / atm.g

        a = np.zeros(4, 'd')
        b = np.zeros(4, 'd')
        c = np.zeros(4, 'd')
    
        for i in range(0, 4):
            dh   = h[i] - h0
            dt   = atm.temperature(h[i]) - atm.temperature(h0)
            ab   = dt/dh * atm.R / (atm.M_m * atm.g)
            c[i] = atm.R * atm.temperature(h0) / (atm.M_m * atm.g) / (1 + ab)
            a[i] = (atm.overburden[i] - X0*np.exp(-dh/c[i]))/(1 - np.exp(-dh/c[i]))
            b[i] = (X0 - a[i])*np.exp(h0/c[i])
            h0   = h[i]
            X0   = atm.overburden[i]
        
        self._par = np.vstack((a, b, c))
        return self
        
    def improveFit(self, atm):
        p0 = self._par.flatten()
        fit_par, fit_pcov = curve_fit(self, atm.h, atm.overburden, p0 = p0)
        self._par = np.array(fit_par)
        self._par.resize((3, 4))
        
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
    
    