import numpy as np
from atm import generate_atmosphere
from concurrent.futures import ProcessPoolExecutor
import mceq_config
from MCEq.core import MCEqRun
import crflux.models as crf

def simulate_atmosphere(a, theta, mceq=None, alt_grid=None, progress=None):
    """
    Execute MCEq simulation of one atmosphere, a, over angles theta.

    Parameters
    ----------
    a : EarthsAtmosphere
        Atmosphere
    theta : float or list
        Zenith angles
    mceq : MCEqRun or None
        MCEq kernel. If None one will be created within this function. This 
        latter case may be useful for multiprocessing, if I can ever get it
        to work.
    alt_grid : list of floats
        Altitude grid to solve fluxes on - must be transformed to an X grid
        by the rotated Atmosphere.
    progress : tqdm or None
        Progress bar a la tqdm
    """
    if mceq is None: 
        mceq = MCEqRun(
            interaction_model='SIBYLL2.3c',
            primary_model = (crf.HillasGaisser2012, 'H3a'),
            theta_deg = 0.
        )

    mceq.set_density_model(a)

    theta = np.atleast_1d(theta)
    flux_A = []
    depths = []
    
    for t in theta:
        mceq.set_theta_deg(t)
        int_grid = None if alt_grid is None else a.h2X(np.array(alt_grid)*100.0)
        mceq.solve(int_grid)
        flux_X = []
        depth  = np.empty(0, 'd')
        if int_grid is not None:
            depth = np.concatenate((depth, int_grid))
            for i in range(len(int_grid)):
                flux_X.append(np.copy(
                    mceq.get_solution('total_mu-', grid_idx=i) + \
                    mceq.get_solution('total_mu+', grid_idx=i)))

        flux_X.append(np.copy(
            mceq.get_solution('total_mu-') + \
            mceq.get_solution('total_mu+')))
        depths.append(np.concatenate((depth, (a.max_X,))))
        flux_A.append(np.stack(flux_X, 0))

        if progress is not None: progress.update(1)

    return a.profile, np.stack(depths, 0), np.swapaxes(np.stack(flux_A, 0), 0, 1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Random atmospheric muon flux generator/simulator'
    )
    parser.add_argument('run', type=int, 
                        help='Run number')
    parser.add_argument('N', type=int, 
                        help='Number of patterns to generate')
    parser.add_argument('-b', '--barometer', type=float, default=29.92)
    parser.add_argument('-t', '--theta', nargs='+', type=float, default=4,
                        help='List of theta angles (degrees) to simulate')
    parser.add_argument('-m', '--multiprocess', metavar='M', type=int, default=1,
                        help='Number of processes to use (default=1 no multiprocessing)')
    parser.add_argument('-p', '--prefix', type=str, default='ATM')
    parser.add_argument('-u', '--progress', action='store_true', default=False,
                        help='Display tqdm progress bar.')
    parser.add_argument('--e-min', metavar='E0', type=float, default=1.0)
    parser.add_argument('--e-max', metavar='E1', type=float, default=1E10)
    parser.add_argument('-H', '--alt-grid', nargs='+', type=float)

    args = parser.parse_args()

    filename = f'{args.prefix}{args.run:06d}.npz'
    theta = np.array(args.theta, 'd')
    mceq_config.e_min = args.e_min
    mceq_config.e_max = args.e_max
    P0 = args.barometer * 3386.53075
    progress = None

    try:
        if args.multiprocess > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.multiprocess) as exe:
                futures = []
                for i in range(args.N):
                    futures.append(exe.submit(
                        simulate_atmosphere, 
                        generate_atmosphere(P0 = P0), theta))

            res = [f.result() for f in futures]
        else:
            mceq = MCEqRun(
                interaction_model='SIBYLL2.3c',
                primary_model = (crf.HillasGaisser2012, 'H3a'),
                theta_deg = 0.
            )
            if args.progress:
                from tqdm import tqdm
                progress = tqdm(total=args.N*len(theta))
        
            res = []
            for i in range(args.N):
                try:
                    a = generate_atmosphere(P0 = P0)
                    res.append(
                        simulate_atmosphere(
                        a, theta, mceq=mceq, 
                        alt_grid=args.alt_grid, 
                        progress=progress))
                except ValueError as err:
                    print(f'Caught Error on Event #{i}:', err)
                    
    except KeyboardInterrupt:
        print(f'Caught interrupt - terminating early with {len(res)} atmospheres.')
    finally:        

        atm, ig, flux = list(zip(*res))
        atm = np.stack(atm, 0)
        ig  = np.stack(ig, 0)
        flux = np.stack(flux, 0)

        np.savez(filename, e_grid=mceq.e_grid, 
                 pressure=np.array((P0,), 'd'),
                 int_grid=ig, alt_grid=np.array(args.alt_grid, 'd'),
                 theta=theta, atm=atm, flux=flux)

