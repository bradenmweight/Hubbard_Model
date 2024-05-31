import numpy as np
from itertools import product, combinations
from matplotlib import pyplot as plt

def get_Globals( tl ):
    # Parameters
    global N, t, U, n_up, n_down
    N      = 4  # Number of sites
    t      = tl  # Hopping parameter
    U      = 1.0  # On-site interaction parameter
    n_up   = 2  # Number of spin-up electrons
    n_down = 2  # Number of spin-down electrons

def create_hubbard_basis(N, n_up, n_down):
    """
    Create the basis states for the Hubbard model with N sites,
    n_up spin-up electrons, and n_down spin-down electrons.
    """
    global NBASIS, basis_states

    sites = range(N)
    up_states = list(combinations(sites, n_up))
    down_states = list(combinations(sites, n_down))
    
    basis_states = []
    for up in up_states:
        for down in down_states:
            state = [0] * (2 * N)
            for site in up:
                state[site] = 1  # Spin-up electron
            for site in down:
                state[site + N] = 1  # Spin-down electron
            basis_states.append(state)
    
    NBASIS = len(basis_states)

    print(f"dim(H) = {NBASIS}")
    for bi,b in enumerate( basis_states ):
        print(bi, b)

    basis_states = np.array(basis_states)

def create_hubbard_hamiltonian():
    """
    Create the Hamiltonian for the 1D Hubbard model with N sites, t hopping parameter,
    and U on-site interaction, restricted to the given basis states.
    """
    H = np.zeros((NBASIS, NBASIS))
    
    # Hopping term
    for b1 in range( NBASIS-1 ):
        b2 = b1 + 1
        state1 = basis_states[b1]
        state2 = basis_states[b2]
        for i in range(N):
            CHANGE_ALPHA = state1[:N] - state2[:N]
            CHANGE_BETA  = state1[N:] - state2[N:]
            #print(state1, "-->", state2, CHANGE_ALPHA, CHANGE_BETA, np.sum(np.abs(CHANGE_ALPHA))//2, np.sum(np.abs(CHANGE_BETA))//2 )
            if ( np.sum(np.abs(CHANGE_ALPHA))//2 == 0 and np.sum(np.abs(CHANGE_BETA))//2 == 1 ):
                H[b1,b2] = -t
                H[b2,b1] = -t
            elif ( np.sum(np.abs(CHANGE_ALPHA))//2 == 1 and np.sum(np.abs(CHANGE_BETA))//2 == 0 ):
                H[b1,b2] = -t
                H[b2,b1] = -t

        
    # On-site interaction term
    for b in range( NBASIS ):
        state   = basis_states[b]
        OCC     = (state[:N] == 1) * (state[N:] == 1)
        NU      = np.sum( OCC )
        #print(state[:N], state[N:], NU)
        H[b,b]  = NU * U
    

    # plt.imshow( H, origin='lower', cmap='viridis' )
    # plt.colorbar(pad=0.01)
    # plt.xticks( ticks=np.arange(NBASIS), labels=["".join(map(str,basis_states[b])) for b in range(NBASIS)], rotation=60, fontsize=5 )
    # plt.yticks( ticks=np.arange(NBASIS), labels=["".join(map(str,basis_states[b])) for b in range(NBASIS)], rotation=30, fontsize=5 )
    # plt.tight_layout()
    # plt.savefig("H.jpg", dpi=300)
    # plt.clf()
    return H

def plot_Energy( E, t_LIST ):
    for state in range( NBASIS ):
        plt.plot( t_LIST, E[:,state], '-', c='black' ) # , label="$\psi_%1.0f$" % state )
    #plt.legend()
    plt.xlabel("Hopping, t (a.u.)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.title("$U$ = %1.3f a.u.    $N_\mathrm{sites}$ = %1.0f" % (U,N), fontsize=15)
    plt.savefig("E.jpg", dpi=300)
    plt.clf()

def plot_Wavefunctions( E, WFN, t_LIST ):
    # Full basis
    for state in range( 1 ):
        plt.imshow( WFN[:,:,state].T**2, origin='lower', cmap='afmhot_r', extent=[t_LIST[0], t_LIST[-1], 0.5, NBASIS+0.5], aspect='auto' )
        plt.colorbar(pad=0.01)
        #plt.yticks( ticks=np.arange(1, NBASIS+1,1), labels=np.arange(1,NBASIS+1,1) )
        plt.yticks( ticks=np.arange(1, NBASIS+1,1), labels=["".join(map(str,basis_states[b])) for b in range(NBASIS)] )
        plt.xlabel("Hopping Parameter, t (a.u.)", fontsize=15)
        plt.ylabel("Basis State", fontsize=15)
        plt.title("$U$ = %1.3f a.u.    $N_\mathrm{sites}$ = %1.0f" % (U,N), fontsize=15)
        plt.tight_layout()
        plt.savefig(f"WFN_{state}.jpg", dpi=300)
        plt.clf()

    # Project into site basis
    # N_OP = np.diag( np.arange(N) )
    # N_OP = np.kron( np.identity(2), N_OP )
    # print(N_OP)
    # exit()
    #for state in range( NBASIS ):
    for state in range( 1 ):
        POP = np.zeros( (len(t_LIST),N) )
        for s in range( N ):
            for b in range( NBASIS ):
                if ( basis_states[b,s] == 1 or basis_states[b,s+N] == 1 ):
                    #print( "BASIS = ", basis_states[b,:], "Occupation = %1.0f" % (basis_states[b,s] + basis_states[b,s+N]), "on site %s" % s,  )
                    POP[:,s] += WFN[:,b,state]**2 * (basis_states[b,s] + basis_states[b,s+N])
        plt.imshow( POP[:,:].T, origin='lower', cmap='afmhot_r', extent=[t_LIST[0], t_LIST[-1], 0.5, N+0.5], aspect='auto' )
        cbar = plt.colorbar(pad=0.01,label="Average Occupation, $\langle \hat{N} \\rangle$")
        plt.yticks( ticks=np.arange(1, N+1,1), labels=np.arange(1, N+1,1) )
        plt.xlabel("Hopping Parameter, t (a.u.)", fontsize=15)
        plt.ylabel("Lattice Site, $n$", fontsize=15)
        plt.title("$U$ = %1.3f a.u.    $N_\mathrm{sites}$ = %1.0f" % (U,N), fontsize=15)
        plt.tight_layout()
        plt.savefig(f"WFN_{state}_SITE.jpg", dpi=300)
        plt.clf()

def main():

    t_LIST = np.arange( 0.0, 2.0, 0.05 )
    E   = 0
    WFN = 0
    
    for ti, t in enumerate(t_LIST):
        get_Globals(t)

        # Create the basis states with the given electron configuration
        create_hubbard_basis(N, n_up, n_down)
        if ( ti == 0 ):
            E   = np.zeros( (len(t_LIST), NBASIS) )
            WFN = np.zeros( (len(t_LIST), NBASIS, NBASIS) )

        # Create the Hamiltonian
        H = create_hubbard_hamiltonian()

        # Find the ground state energy and eigenvector
        E[ti,:], WFN[ti,:,:] = np.linalg.eigh(H)  # Find all eigenvalues and eigenvectors

    plot_Energy( E, t_LIST )
    plot_Wavefunctions( E, WFN, t_LIST )


if ( __name__ == "__main__"):
    main()