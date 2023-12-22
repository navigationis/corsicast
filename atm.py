import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import solve_ivp
from datetime import datetime

from MCEq.geometry.density_profiles import EarthsAtmosphere

class Rawinsonde:
    """
    Class for handling rawinsondes or radiosondes data.
    [NCEI](https://www.ncei.noaa.gov/products/weather-balloon/integrated-global-radiosonde-archive) 
    archives data which can be directly read by :func:read_station_data. 
    Normally you should not need to instantiate this class directly but rather 
    retrieve observations from the archive.
    """
    def __init__(self, hdr:str):
        self.st_id = hdr[1:13]
        self.dtime = datetime(int(hdr[13:17]), int(hdr[18:20]), int(hdr[21:23]), int(hdr[24:26]))
        self.numlev = int(hdr[32:36])
        self.p_src  = hdr[37:45]
        self.np_src = hdr[46:54]
        self.lat    = float(hdr[55:62])/10000.
        self.lon    = float(hdr[63:71])/10000.
        self.obs = np.zeros(self.numlev, dtype=[('lvltyp', 'i8'), ('etime', 'i4'),
            ('pressure', np.float32), ('gph', np.float32), ('temperature', np.float32), 
            ('rh', np.float32), ('dewpoint', np.float32), ('winddir', np.float32),
            ('windspeed', np.float32)])
        self.__iobs = 0
        self.invalid  = None

    def add_observation(self, obstxt:str):
        i = self.__iobs
        self.__iobs += 1
        self.obs[i] = (int(obstxt[0:2]), int(obstxt[3:8]), float(obstxt[9:15])/100.,
            float(obstxt[16:21]), float(obstxt[22:27])/10., float(obstxt[28:33])/10.,
            float(obstxt[34:39])/10., float(obstxt[40:45]), float(obstxt[46:51]))
        if self.__iobs == self.numlev: self.invalid = (self.obs['pressure'] < 0.0) | \
            (self.obs['temperature'] < -100.0) | \
            (self.obs['gph'] == -9999.0)

    @property
    def observation(self):
        return self.obs.view(np.recarray)

    @property
    def pressure(self):
        return ma.masked_array(self.obs['pressure'], mask=self.obs['pressure']<0.0)

    @property
    def temperature(self):
        return ma.masked_array(self.obs['temperature'], mask=self.obs['temperature']==-999.9)

    @property
    def geopotential_height(self):
        return ma.masked_array(self.obs['gph'], mask=self.obs['pressure']<0.0)

    @property
    def density(self):
        return 100.0 * self.pressure * 0.028966 / (8.314 * (self.temperature + 273.16))
    
    @property 
    def valid(self):
        return np.rec.array(self.obs[~self.invalid])
    
def read_station_data(filename: str) -> Rawinsonde:
    """
    Read [NCEI](https://www.ncei.noaa.gov/products/weather-balloon/integrated-global-radiosonde-archive) 
    radiosonde data archives.
    """
    raw = None
    i = 0
    with open(filename, 'rt') as f:
        while True:
            i += 1
            line = f.readline()
            if len(line) == 0: return raw
            if line[0] == '#':
                if raw is not None: yield raw
                raw = Rawinsonde(line)
            else:
                raw.add_observation(line)

def dPdh(h, p, temperature):
    M_m = 0.028966 
    R   = 8.314
    g   = 9.80
    return -p * M_m * g / (R * temperature(h))
    
class RWSAtmosphere(EarthsAtmosphere):
    def __init__(self, rws_obs):
        super().__init__(self)
        self.rws = rws_obs

        # Setup the atmospheric density calculator
        gph = self.rws.valid.gph
        tmp = self.rws.valid.temperature
        P0  = self.rws.valid.pressure[0] * 100
        upper_z = np.array((48000, 54000, 61000, 80000, 90000, 100000), 'd')
        upper_T = np.array((0, 0, -10, -80, -80, -65), 'd')
        z = np.concatenate((gph, upper_z))
        T = np.concatenate((tmp, upper_T)) + 273.16
        self._spline_T = UnivariateSpline(z, T, s=25, ext='const')
        t_eval = np.concatenate((np.arange(0, 10000, 250, dtype='d'),
                                 np.arange(10000, 30000, 500, dtype='d'),
                                 10**np.linspace(4.475, 5, 50)))
        self.psoln = solve_ivp(dPdh, (0, 1E5), y0=(P0,), t_eval = t_eval, args=(self._spline_T,))
        self._spline_P = UnivariateSpline(self.psoln.t, self.psoln.y[0], k=1, s=0, ext='const')

    def get_density(self, h_cm):
        z = h_cm / 100.0
        P = self._spline_P(z)
        T = self._spline_T(z)
        return P * 0.028966 / (8.314 * T) * 0.001
    
class RWSim:
    def __init__(self):
        self.obs = np.zeros(5, dtype=[('gph', 'd'), ('temperature', 'd'), ('pressure', 'd')])

        a0 = np.random.uniform(0, 1)
        a1 = np.random.exponential(scale=150)
        a2 = np.random.normal(loc=1250, scale=250)
        a3 = np.random.uniform(0, 3000)
        
        self.obs['gph'][0] = np.where(a0 > 0.95, a3, np.where(a0 > 0.88, a2, a1))
        self.SLP = np.random.normal(loc=29.92, scale=0.5)
    
        self.obs['pressure'][0] = self.SLP * 33.87 * ((288.16 - 0.0065*self.obs['gph'][0])/288.16)**5.26
        
        self.obs['temperature'][0] = np.random.weibull(12)*85 - 60

        lapse_1 = -0.0065 + np.random.uniform(-0.0015, 0.0015)
        self.obs['gph'][1]  = np.random.uniform(8000, 15000)
        self.obs['temperature'][1] = self.obs['temperature'][0] + self.obs['gph'][1]*lapse_1

        lapse_2 = np.random.uniform(-0.0005, 0.0005)
        self.obs['gph'][2]  = self.obs['gph'][1] + np.random.uniform(0, 5000)
        self.obs['temperature'][2] = self.obs['temperature'][1] + self.obs['gph'][2]*lapse_2

        lapse_3 = np.random.uniform(-0.001, 0.001)
        self.obs['gph'][3]  = self.obs['gph'][2] + np.random.uniform(2000, 5000)
        self.obs['temperature'][3] = self.obs['temperature'][2] + self.obs['gph'][3]*lapse_3

        lapse_4 = np.random.uniform(0.00025, 0.002)
        self.obs['gph'][4]  = 35000.0
        self.obs['temperature'][4] = self.obs['temperature'][3] + (self.obs['gph'][4] - self.obs['gph'][3])*lapse_4

        self.valid = np.rec.array(self.obs)
