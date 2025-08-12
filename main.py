import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Estados
SUSC = 0
INF = 1
REC = 2


if __name__ == "__main__":
    
    SIZE = (120, 120)
    PROB_INFECT = 0.25      # probabilidade de transmissão por vizinho por passo
    PROB_RECOVER = 0.02     # probabilidade de recuperação por passo (se infectious_period=None)
    INITIAL_INF = 0.002     # fração inicial infectada
    MOBILITY = 0.01         # fração de swaps por passo
    STEPS = 600

    sim = SIRGrid(size=SIZE,
                  prob_infect=PROB_INFECT,
                  prob_recover=PROB_RECOVER,
                  initial_infected=INITIAL_INF,
                  infectious_period=None,  # ou um inteiro, ex: 14
                  mobility=MOBILITY,
                  seed=42)

    sim.run(max_steps=STEPS)
    sim.plot_curves()

    # Mostrar o estado final como imagem
    plt.figure(figsize=(6,6))
    plt.imshow(sim.grid, vmin=0, vmax=2)
    plt.title(f"Estado final t={sim.t}")
    plt.axis('off')
    plt.show()

