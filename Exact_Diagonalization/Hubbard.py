import numpy as np
from itertools import product, combinations
from matplotlib import pyplot as plt

def get_Globals( tl ):
    # Parameters
    global N, t, U, n_up, n_down#, PERIODIC
    N        = 4  # Number of sites
    t        = tl  # Hopping parameter
    U        = 1.0  # On-site interaction parameter
    n_up     = 1  # Number of spin-up electrons
    n_down   = 1  # Number of spin-down electrons
    #PERIODIC = True # TODO

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

    np.savetxt("BASIS_STATES.dat", basis_states, fmt="%1.0f")

    #print(f"dim(H) = {NBASIS}")
    #for bi,b in enumerate( basis_states ):
    #    print(bi, b)

    basis_states = np.array(basis_states)

def create_hubbard_hamiltonian():
    """
    Create the Hamiltonian for the 1D Hubbard model with N sites, t hopping parameter,
    and U on-site interaction, restricted to the given basis states.
    """
    H = np.zeros((NBASIS, NBASIS))
    
    # Hopping term
    for b1 in range( NBASIS ):
        for b2 in range( NBASIS ):
            state1 = basis_states[b1]
            state2 = basis_states[b2]
            if ( (state1[N:] == state2[N:]).all() ): # Check for frozen spin-down
                ZEROS = (state1[:N] + state2[:N] - 1 == 0)
                ZEROS = np.where(ZEROS)[0]
                if ( len(ZEROS) == 2 ):
                    if ( abs(ZEROS[1] - ZEROS[0]) == 1 ):
                        #print("UP")
                        #print("   ALPHA", state1[:N], state2[:N])
                        #print("   BETA ", state1[N:], state2[N:])
                        H[b1,b2] = -t
                        H[b2,b1] = -t
            elif ( (state1[:N] == state2[:N]).all() ): # Check for frozen spin-up
                ZEROS = (state1[N:] + state2[N:] - 1 == 0)
                ZEROS = np.where(ZEROS)[0]
                if ( len(ZEROS) == 2 ):
                    if ( abs(ZEROS[1] - ZEROS[0]) == 1 ):
                        #print("UP")
                        #print("   ALPHA", state1[:N], state2[:N])
                        #print("   BETA ", state1[N:], state2[N:])
                        H[b1,b2] = -t
                        H[b2,b1] = -t



        
    # On-site interaction term
    for b in range( NBASIS ):
        state   = basis_states[b]
        OCC     = (state[:N] == 1) * (state[N:] == 1)
        NU      = np.sum( OCC )
        #print(state[:N], state[N:], NU)
        H[b,b]  = NU * U

    return H

def plot_Energy( E, t_LIST ):
    for state in range( NBASIS ):
        plt.plot( t_LIST/U, E[:,state]/U, '-', c='black' ) # , label="$\psi_%1.0f$" % state )
    #plt.legend()
    plt.xlabel("t / U", fontsize=15)
    plt.ylabel("E / U", fontsize=15)
    plt.xlim(t_LIST[0]/U, t_LIST[-1]/U)
    plt.title("$U$ = %1.3f a.u.    $N_\mathrm{sites}$ = %1.0f    $N_\mathrm{up}$ = %1.0f    $N_\mathrm{down}$ = %1.0f" % (U,N,n_up,n_down), fontsize=15)
    plt.tight_layout()
    plt.savefig("E.jpg", dpi=300)
    plt.clf()

    for state in range( NBASIS ):
        plt.plot( t_LIST/U, E[:,state]/U - E[:,0]/U, '-', c='black' ) # , label="$\psi_%1.0f$" % state )
    #plt.legend()
    plt.xlabel("t / U", fontsize=15)
    plt.ylabel("E / U", fontsize=15)
    plt.xlim(t_LIST[0]/U, t_LIST[-1]/U)
    plt.title("$U$ = %1.3f a.u.    $N_\mathrm{sites}$ = %1.0f    $N_\mathrm{up}$ = %1.0f    $N_\mathrm{down}$ = %1.0f" % (U,N,n_up,n_down), fontsize=15)
    plt.tight_layout()
    plt.savefig("E_TRANS.jpg", dpi=300)
    plt.clf()

def plot_Wavefunctions( E, WFN, t_LIST ):
    # Full basis
    for state in range( 1 ):
        plt.imshow( WFN[:,:,state].T**2, origin='lower', cmap='afmhot_r', extent=[t_LIST[0]/U, t_LIST[-1]/U, 0.5, NBASIS+0.5], aspect='auto' )
        plt.colorbar(pad=0.01)
        #plt.yticks( ticks=np.arange(1, NBASIS+1,1), labels=np.arange(1,NBASIS+1,1) )
        if ( NBASIS <= 100 ):
            plt.yticks( ticks=np.arange(1, NBASIS+1,1), labels=["".join(map(str,basis_states[b])) for b in range(NBASIS)] )
        plt.xlabel("t / U", fontsize=15)
        plt.ylabel("Basis State", fontsize=15)
        plt.title("$U$ = %1.3f a.u.    $N_\mathrm{sites}$ = %1.0f    $N_\mathrm{up}$ = %1.0f    $N_\mathrm{down}$ = %1.0f" % (U,N,n_up,n_down), fontsize=15)
        plt.tight_layout()
        plt.savefig(f"WFN_{state}.jpg", dpi=300)
        plt.clf()

    # Project into site basis
    for state in range( 1 ):
        POP = np.zeros( (len(t_LIST),N) )
        for s in range( N ):
            for b in range( NBASIS ):
                if ( basis_states[b,s] == 1 or basis_states[b,s+N] == 1 ):
                    #print( "BASIS = ", basis_states[b,:], "Occupation = %1.0f" % (basis_states[b,s] + basis_states[b,s+N]), "on site %s" % s,  )
                    #POP[:,s] += WFN[:,b,state]**2 * (basis_states[b,s] + basis_states[b,s+N]) # Calculate Average Electron Occupation to Each Site
                    POP[:,s] += WFN[:,b,state]**2 # Calculate Site Contribution
        plt.imshow( POP[:,:].T, origin='lower', cmap='afmhot_r', extent=[t_LIST[0]/U, t_LIST[-1]/U, 0.5, N+0.5], aspect='auto' )
        #plt.colorbar(pad=0.01,label="Average Occupation, $\langle \hat{N} \\rangle$")
        plt.colorbar(pad=0.01,label="Site Population, $\langle \hat{N} \\rangle$")
        if ( NBASIS <= 100 ):
            plt.yticks( ticks=np.arange(1, N+1,1), labels=np.arange(1, N+1,1) )
        plt.xlabel("t / U", fontsize=15)
        plt.ylabel("Lattice Site, $n$", fontsize=15)
        plt.title("$U$ = %1.3f a.u.    $N_\mathrm{sites}$ = %1.0f    $N_\mathrm{up}$ = %1.0f    $N_\mathrm{down}$ = %1.0f" % (U,N,n_up,n_down), fontsize=15)
        plt.tight_layout()
        plt.savefig(f"WFN_{state}_SITE.jpg", dpi=300)
        plt.clf()

def plot_Hamiltonian( H ):

    plt.imshow( H/U, origin='lower', cmap='viridis', vmin=-(n_up + n_down)//2, vmax=(n_up + n_down)//2 )
    plt.colorbar(pad=0.01)
    if ( NBASIS <= 100 ):
        plt.xticks( ticks=np.arange(NBASIS), labels=["".join(map(str,basis_states[b])) for b in range(NBASIS)], rotation=90, fontsize=5 )
        plt.yticks( ticks=np.arange(NBASIS), labels=["".join(map(str,basis_states[b])) for b in range(NBASIS)], rotation=0, fontsize=5 )
    plt.tight_layout()
    plt.savefig("H.jpg", dpi=300)
    plt.clf()

def plot_IPR( WFN, t_LIST ):
    """
    IPR = (\sum_j^N WFN_j**4) * (1 / \sum_k^N WFN_k**2)
    """

    # Project into site basis
    POP = np.zeros( (len(t_LIST),N) )
    IPR = np.zeros( (len(t_LIST),NBASIS) )
    for state in range( NBASIS ):
        for b in range( NBASIS ):
            for s in range( N ):
                if ( basis_states[b,s] == 1 or basis_states[b,s+N] == 1 ):
                    POP[:,s] += WFN[:,b,state]**2 #* (basis_states[b,s] + basis_states[b,s+N]) # Calculate Site Population
        POP = POP / np.sum( POP[:,:], axis=-1 )[:,None]
        IPR[:,state] = 1 / np.sum( POP[:,:]**2, axis=-1 ) #/ np.sum( POP[:,:], axis=-1 )
    plt.imshow( IPR[:,:].T, origin='lower', cmap='afmhot_r', extent=[t_LIST[0]/U, t_LIST[-1]/U, 0.5, NBASIS+0.5], aspect='auto' )
    plt.colorbar(pad=0.01,label="IPR (\# of Sites)")
    if ( NBASIS <= 100 ):
        plt.yticks( ticks=np.arange(1, NBASIS+1,1), labels=np.arange(1, NBASIS+1,1) )
    plt.xlabel("t / U", fontsize=15)
    plt.ylabel("State, $j$", fontsize=15)
    plt.title("$U$ = %1.3f a.u.    $N_\mathrm{sites}$ = %1.0f    $N_\mathrm{up}$ = %1.0f    $N_\mathrm{down}$ = %1.0f" % (U,N,n_up,n_down), fontsize=15)
    plt.tight_layout()
    plt.savefig(f"IPR.jpg", dpi=300)
    plt.clf()

def main():

    #t_LIST = np.array([0.0, 1.0])
    t_LIST = np.linspace( 0.0, 1.0, 51 )
    E   = 0
    WFN = 0
    
    for ti, t in enumerate(t_LIST):
        get_Globals(t)

        # Create the basis states with the given electron configuration
        create_hubbard_basis(N, n_up, n_down)
        print(ti, t, NBASIS)
        if ( ti == 0 ):
            E   = np.zeros( (len(t_LIST), NBASIS) )
            WFN = np.zeros( (len(t_LIST), NBASIS, NBASIS) )

        # Create the Hamiltonian
        H = create_hubbard_hamiltonian()
        if ( ti == len(t_LIST)-1 ):
            plot_Hamiltonian( H )

        # Find the ground state energy and eigenvector
        E[ti,:], WFN[ti,:,:] = np.linalg.eigh(H)  # Find all eigenvalues and eigenvectors

    plt.plot( WFN[-1,:,0]**2, lw=3 )
    plt.plot( WFN[-1,:,1]**2, "--", lw=1.5 )
    plt.tight_layout()
    plt.savefig("TEST.jpg", dpi=300)
    plt.clf()

    plot_Energy( E, t_LIST )
    plot_Wavefunctions( E, WFN, t_LIST )
    plot_IPR( WFN, t_LIST )



if ( __name__ == "__main__"):
    main()