"""
Atmosphere module.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import RK23, OdeSolution
from datetime import datetime

from MCEq.geometry.density_profiles import EarthsAtmosphere

M_m = 0.028966  
R   = 8.314
g   = 9.80665   

def scale_height(T_C):
    """
    Return isothermal scale height in km given a temperature
    """
    return 8.314 * (T_C + 273.16) / (0.029 * 9.8) * 0.001

def dPdh(h, p, temperature):
    """
    Change of pressure with respect to height. Units are MKS.
    Used for constructing atmospheres from temperature profiles
    through solution of the differential equation though :func:`scipy.integrate.solve_ivp`

    Parameters
    ----------
    h : float or numpy.ndarray-like
        Geopotential height in meters.
    p : float 
        Pressure
    temperature : {function}
        Function that provide temperature, in Kelvin, versus geopotential height
    """
    return -p * M_m * g / (R * temperature(h))

class AtmospherefromTemperatureProfile(EarthsAtmosphere):
    """
    Atmosphere created from temperature profile.

    Parameters
    ----------
    alt : numpy.ndarray or list or float
        Geopotential altitudes, in meters, of temperature points.
    T : numpy.ndarray or list or float
        Temperature, in °C, at sample points.
    P0 : float
        Reference pressure, in Pa.
    isa_fill : bool
        If True, fill missing values above maximum altitude specified
        in alt from the ISA standard values.
    spline_k : int
        Degree of spline interpolation for T spline
    spline_s : float
        Smoothness parameter for spline interpolation

    Notes
    -----
    If `alt` or `T` are omitted and isa_fill is set to True then the
    atmosphere created is the International Standard Atmosphere (ISA).

    Examples
    --------
    Define the temperature at a few points in the troposphere and let
    the constructor fill in the ISA above (in this case beginning at 11 km):
    >>> z = (1000, 2000, 10000)
    >>> T = (10, 5, -20)
    >>> a = atm.fromTemperatureProfile(alt=z, T=T, P0=90000., isa_fill=True)

    One possible use case where one desires to use the ISA for most of the
    atmosphere but perhaps wants to change the sea level temperature and 
    pressure (here 30.00" Hg):
    >>> a = atm.fromTemperatureProfile(T=25, P0=101596, isa_fill=True)
    """
    _isa = np.array((
        ( -610,  19.0), (11000, -56.5), (20000, -56.5), 
        (32000, -44.5), (47000,  -2.5), (51000,  -2.5),
        (71000, -58.5), (84852, -86.28)), dtype = 'd')
    
    def __init__(
            self, 
            alt: list = None, 
            T: list = None, 
            P0: float = 108900.,
            isa_fill: bool = False, 
            spline_k: int = 1,
            spline_s: float = 0):
        
        super().__init__(self)

        if T is None:
            if not isa_fill: raise ValueError('Empty temperature profile')
            self.profile = self._isa
        else:
            if alt is None: alt = 0.0
            alt = np.atleast_1d(alt)
            T   = np.atleast_1d(T)
            self.profile = np.stack((alt, T), -1)
            if isa_fill: 
                self.profile = np.concatenate((
                    self.profile,
                    np.array(list(filter(lambda x: x[0] > alt[-1], self._isa)))
                    ))
                
        self.P0 = P0
        self._Tspline = UnivariateSpline(
            self.profile[:,0], self.profile[:,1] + 273.16,
            k = spline_k, s = spline_s, ext = 'const')
        
        z0 = self.profile[0,0]
        rk = RK23(lambda h, p: dPdh(h, p, self._Tspline), 
                  z0, np.atleast_1d(self.P0), 100000.0)
        
        z = [z0]
        interpolants = []
        while rk.status == 'running':
            rk.step()
            z.append(rk.t)
            interpolants.append(rk.dense_output())

        self._p = (z, interpolants)
        
        self._pressure_solution = OdeSolution(z, interpolants)

    def pressure(self, alt: float | list | np.ndarray):
        """
        Return pressure, in Pascal, given altitude in meters.

        Parameters
        ----------
        alt : float or list of floats
            Altitude in meters

        Returns
        -------
        pa : float or list of floats
            Pressure, in Pa.
        """
        return self._pressure_solution(alt).flatten()
    
    def temperature(self, alt: float | list | np.ndarray):
        """ 
        Return temperature, in Kelvin, given altitude in meters.

        Parameters
        ----------
        alt : float or list of floats
            Altitude in meters.

        Returns
        -------
        T : float or list of floats
            Temperature, in K.
        """
        return self._Tspline(alt)
    
    def thickness(self, alt: float):
        """ 
        Return thickness, in kg/m^2.
        """
        return self.pressure(alt) / g
    
    def density(self, alt: float):
        """ 
        Return density, in kg/m^3
        """
        return self.pressure(alt) * 0.028966 / (8.314 * self._Tspline(alt))
    
    def get_density(self, h_cm: float) -> float:
        """
        Overriden density function from MCEq.geometry.density_profiles.EarthsAtmosphere.
        Returns density in g/cm^3 units given a height h_cm in centimeters.

        Parameters
        ----------
        h_cm : float
            Height, in cm.

        Returns
        -------
        rho : float
            Density at h_cm, in g/cm^3
        """
        z = h_cm / 100.0
        if z > 100000.0: return 1E-15
        return self.density(z) * 0.001


def generate_atmosphere(
        alt: float = 0.0, 
        P0: float = 101325.,
        temp_a: float = 8.0,
        temp_b: float = 85.):
    """
    Create Atmosphere by generating random temperature profile.

    Parameters
    ----------
    alt : float
        Base (i.e. ground level) altitude, in meters.
    P0 : float
        Ground level pressure, in Pa
    temp_a: float
        Weibull a parameter for ground level temperature randomization.
        Smaller numbers will broaden the distribution.
    temp_b: float
        Weibull width parameter for ground level temperature randomization.
        Larger numbers will broaden the distribution.
    """
    atm_par = (((  500, 1000), (-0.0075, 0.0025)), # troposphere - 1
               (( 2500, 8000), (-0.0080,-0.0045)), # troposphere - 2
               (( 4000, 9000), (-0.0080,-0.0045)), # troposphere - 3
               (( 1000, 4000), (-0.0050, 0.0025)), # troposphere - 4
               ((  500,15000), (-0.0010, 0.0010)), # tropopause - 1
               ((12000,20000), ( 0.0000, 0.0035)), # stratosphere - 1
               (( 1000, 8000), (-0.0002, 0.0002)), # stratopause
               ((15000,25000), (-0.0025,-0.0020)), # mesosphere - 1
               (( 5000,10000), (-0.0001, 0.0001)), # mesopause
               ((10000,20000), (-0.0001, 0.0035)), # thermosphere
               ((10000,20000), ( 0.0000, 0.0045)))
    
    z = [alt]
    T = [np.random.weibull(temp_a)*temp_b - 60]

    # build up the atmosphere - first layer might have a temperature inversion
    for (dz0, dz1), (dTdz0, dTdz1) in atm_par:
        z.append(z[-1] + np.random.uniform(dz0, dz1))
        dz = z[-1] - z[-2]
        T.append(T[-1] + np.random.uniform(dTdz0, dTdz1)*dz)

    return AtmospherefromTemperatureProfile(z, T, P0 = P0)

def read_igra_station_list(filename:str):
    """
    Reader for the IGRA v2.2 station list (updated daily).
    Returns a record array encapsulating data in the IGRA station list.
    """
    dtype = [('station_id', 'U11'), ('latitude', float), ('longitude', float), 
             ('elevation', float), ('state', 'U2'), ('name', 'U29'), 
             ('firstyear', int), ('lastyear', int), ('nobs', int)]
    with open(filename, 'rt') as f:
        ra = []
        for line in f.readlines():
            ra.append((line[0:11], float(line[12:20]), float(line[21:30]), 
                       float(line[31:37]), line[38:40], line[41:71], 
                       int(line[72:76]), int(line[77:81]), int(line[82:88])))
    return np.array(ra, dtype)

class Rawinsonde:
    """
    Class for handling rawinsondes or radiosondes data.
    [NCEI](https://www.ncei.noaa.gov/products/weather-balloon/integrated-global-radiosonde-archive) 
    archives data which can be directly read by `read_station_data`. 
    Normally you should not need to instantiate this class directly but rather 
    retrieve observations from the archive using `read_station_data` or
    generate simulated observations.
    """
    _dtype=[('lvltyp', 'i2'), ('etime', 'i4'),
            ('pressure', np.float32), ('gph', np.float32), ('temperature', np.float32), 
            ('rh', np.float32), ('dewpoint', np.float32), ('winddir', np.float32),
            ('windspeed', np.float32)]
    
    def __init__(self, hdr: str):
        self.st_id = hdr[1:13]
        self.dtime = datetime(int(hdr[13:17]), int(hdr[18:20]), int(hdr[21:23]), int(hdr[24:26]))
        self.numlev = int(hdr[32:36])
        self.p_src  = hdr[37:45]
        self.np_src = hdr[46:54]
        self.lat    = float(hdr[55:62])/10000.
        self.lon    = float(hdr[63:71])/10000.
        self.obs = np.zeros(self.numlev, dtype=Rawinsonde._dtype)
        self.__iobs = 0
        self.invalid  = None

    def add_observation(self, obstxt: str):
        i = self.__iobs
        self.__iobs += 1
        self.obs[i] = (int(obstxt[0:2]), int(obstxt[3:8]), float(obstxt[9:15])/100.,
            float(obstxt[16:21]), float(obstxt[22:27])/10., float(obstxt[28:33])/10.,
            float(obstxt[34:39])/10., float(obstxt[40:45]), float(obstxt[46:51]))
        if self.__iobs == self.numlev: self.invalid = (self.obs['pressure'] < 0.0) | \
            (self.obs['temperature'] < -100.0) | \
            (self.obs['gph'] == -9999.0)

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

    Parameters
    ----------
    filename : str
        Path to raw text file holding observation data.
        
    Returns
    -------
    rws : Rawinsonde

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

def generate_station_data(
        n: int,
        st_id: str,
        alt: float | None = None,
        SLP: float | None = None
        ):
    """
    Simulate rawinsonde observations.

    Paramters
    ---------
    n : int
        Number of observations.
    st_id : str
        Station ID.
        Can be any string, IGRA format is not enforced.
    alt : float
        Station altitude

    

    """
    hdr = f"#{id:11s} {year:4d} {month:2d} {day:2d} {hour:2d} 9999" + \
        f" 0005 ncdc-nws ncdc-nws {int(lat*10000):7d} {int(lon*10000):7d}"
    return Rawinsonde(hdr)
