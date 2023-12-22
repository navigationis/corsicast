import numpy as np
from atm import RWSim, RWSAtmosphere
from concurrent.futures import ProcessPoolExecutor
import mceq_config
from MCEq.core import MCEqRun
import crflux.models as crf

def simulate_atmosphere(r, theta, mceq=None):

    if mceq is None: 
        mceq = MCEqRun(
            interaction_model='SIBYLL2.3c',
            primary_model = (crf.HillasGaisser2012, 'H3a'),
            theta_deg = 0.
        )
    theta = np.atleast_1d(theta)
    fluxes = []
    try:
        for t in theta:
            mceq.set_density_model(RWSAtmosphere(r))
            mceq.set_theta_deg(t)
            mceq.solve()
            fluxes.append(np.copy(mceq.get_solution('total_mu-') + mceq.get_solution('total_mu+')))
        fluxes = np.stack(fluxes, -1)
        return r.obs, theta, fluxes
    except ValueError as err:
        return r.obs, theta, err
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Random atmospheric muon flux generator/simulator'
    )
    parser.add_argument('run', type=int, 
                        help='Run number')
    parser.add_argument('N', type=int, 
                        help='Number of patterns to generate')
    parser.add_argument('-a', '--num-angles', metavar='NA', type=int, default=4,
                        help='Number of angular segments between 0 and 90 inclusive')
    parser.add_argument('-m', '--multiprocess', metavar='M', type=int, default=1,
                        help='Number of processes to use (default=1 no multiprocessing)')
    parser.add_argument('-p', '--prefix', type=str, default='ATM')
    parser.add_argument('--e-min', metavar='E0', type=float, default=1.0)
    parser.add_argument('--e-max', metavar='E1', type=float, default=1E10)

    args = parser.parse_args()

    filename = f'{args.prefix}{args.run:06d}.npz'
    theta = np.linspace(0, 90, args.num_angles, 'd')
    mceq_config.e_min = args.e_min
    mceq_config.e_max = args.e_max

    if args.multiprocess > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.multiprocess) as exe:
            futures = []
            for i in range(args.N):
                futures.append(exe.submit(simulate_atmosphere, RWSim(), theta))
    else:
        mceq = MCEqRun(
            interaction_model='SIBYLL2.3c',
            primary_model = (crf.HillasGaisser2012, 'H3a'),
            theta_deg = 0.
        )
        futures = [simulate_atmosphere(RWSim(), theta, mceq=mceq) for i in range(args.N)]

    atm, theta, flux = list(zip(*futures))
    atm = np.stack(atm, -1)
    flux = np.stack(flux, -1)
    np.savez(filename, e_grid=mceq.e_grid, theta=theta[0], atm=atm, flux=flux)

