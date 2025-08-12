class SIRGrid:
    def __init__(self, size=(100,100), prob_infect=0.3, prob_recover=0.05,
                 initial_infected=0.001, infectious_period=None, mobility=0.0, seed=None):
        """
        size: tupla (rows, cols)
        prob_infect: probabilidade de transmissão por contato por passo (beta_local)
        prob_recover: probabilidade de recuperação por passo (gamma) se infectious_period is None
        initial_infected: fração inicial infectada (0-1) ou número absoluto se >=1
        infectious_period: se int N, infectado permanece N passos e então recupera (determinístico)
        mobility: fração (0-1) de pares aleatórios que trocam de posição a cada passo
        seed: seed RNG
        """
        self.rng = np.random.default_rng(seed)
        self.rows, self.cols = size
        self.prob_infect = float(prob_infect)
        self.prob_recover = float(prob_recover)
        self.infectious_period = infectious_period
        self.mobility = float(mobility)

        # criar grid e, se infectious_period, contador de tempo infectado
        self.grid = np.full((self.rows, self.cols), SUSC, dtype=np.int8)
        if infectious_period is not None:
            self.infected_time = np.zeros((self.rows, self.cols), dtype=np.int16)
        else:
            self.infected_time = None

        # inicializar infectados
        total = self.rows * self.cols
        if 0 < initial_infected < 1:
            n_init = int(total * initial_infected)
        else:
            n_init = int(initial_infected)
        n_init = max(1, n_init)  # pelo menos 1
        idx = self.rng.choice(total, size=n_init, replace=False)
        self.grid.flat[idx] = INF
        if self.infected_time is not None:
            self.infected_time.flat[idx] = 0

        # histórico
        self.history = {'S': [], 'I': [], 'R': []}
        self.t = 0

    def neighbors_infected_count(self):
        """Conta vizinhos infectados (8-vizinhos) para cada célula."""
        g = (self.grid == INF).astype(np.int8)
        # usar convolução manual com roll para evitar dependência externa
        s = np.zeros_like(g)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                s += np.roll(np.roll(g, dr, axis=0), dc, axis=1)
        return s

    def step(self):
        """Executa um passo de tempo da simulação."""
        # mobilidade: trocar pares aleatórios
        if self.mobility > 0:
            self.apply_mobility()

        infected_neighbors = self.neighbors_infected_count()
        new_grid = self.grid.copy()

        # Suscetíveis podem ser infectados
        susc_mask = (self.grid == SUSC)
        k = infected_neighbors[susc_mask]
        if k.size > 0:
            # prob de não infectar por k vizinhos = (1 - prob_infect) ** k
            p_not = (1.0 - self.prob_infect) ** k
            p_infect = 1.0 - p_not
            rand = self.rng.random(size=p_infect.shape)
            will_be_infected = rand < p_infect
            # aplicar no new_grid
            coords = np.where(susc_mask)
            infected_coords = (coords[0][will_be_infected], coords[1][will_be_infected])
            new_grid[infected_coords] = INF
            if self.infected_time is not None:
                self.infected_time[infected_coords] = 0

        # Infectados podem recuperar
        inf_mask = (self.grid == INF)
        if self.infectious_period is None:
            # recuperação estocástica
            rand = self.rng.random(size=inf_mask.sum())
            to_recover = rand < self.prob_recover
            coords = np.where(inf_mask)
            recovered_coords = (coords[0][to_recover], coords[1][to_recover])
            new_grid[recovered_coords] = REC
            if self.infected_time is not None:
                self.infected_time[recovered_coords] = 0
        else:
            # contagem de tempo
            self.infected_time[inf_mask] += 1
            # aqueles cujo tempo >= infectious_period recuperam
            to_rec = self.infected_time >= self.infectious_period
            new_grid[to_rec] = REC
            self.infected_time[to_rec] = 0

        self.grid = new_grid
        self.t += 1
        self.record_history()

    def apply_mobility(self):
        """Aplica mobilidade trocando posições de pares aleatórios na grade."""
        total = self.rows * self.cols
        n_swaps = int(self.mobility * total / 2)  # pares
        if n_swaps <= 0:
            return
        idx = self.rng.choice(total, size=2*n_swaps, replace=False)
        # trocar pares (0,1), (2,3), ...
        idx1 = idx[0::2]
        idx2 = idx[1::2]
        flat = self.grid.flat
        a = flat[idx1].copy()
        b = flat[idx2].copy()
        flat[idx1] = b
        flat[idx2] = a
        if self.infected_time is not None:
            ft = self.infected_time.flat
            ta = ft[idx1].copy()
            tb = ft[idx2].copy()
            ft[idx1] = tb
            ft[idx2] = ta

    def record_history(self):
        unique, counts = np.unique(self.grid, return_counts=True)
        cS = cI = cR = 0
        for u, cnt in zip(unique, counts):
            if u == SUSC: cS = cnt
            elif u == INF: cI = cnt
            elif u == REC: cR = cnt
        total = self.rows * self.cols
        self.history['S'].append(cS / total)
        self.history['I'].append(cI / total)
        self.history['R'].append(cR / total)

    def run(self, max_steps=500, stop_if_extinct=True):
        """Roda a simulação até max_steps ou até não haver infectados (se stop_if_extinct)."""
        # gravar estado inicial
        self.record_history()
        for _ in range(max_steps):
            if stop_if_extinct and (self.history['I'] and self.history['I'][-1] == 0):
                break
            self.step()

    # --- visualização ---
    def plot_curves(self, figsize=(8,4), show=True, savepath=None):
        t = np.arange(len(self.history['S']))
        plt.figure(figsize=figsize)
        plt.plot(t, self.history['S'], label='Suscetível')
        plt.plot(t, self.history['I'], label='Infectado')
        plt.plot(t, self.history['R'], label='Recuperado')
        plt.xlabel('Passos de tempo')
        plt.ylabel('Fração da população')
        plt.legend()
        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, dpi=200)
        if show:
            plt.show()
        else:
            plt.close()

    def animate(self, interval=100, frames=None, cmap='viridis', savepath=None):
        """
        Anima a grade (mostrando S/I/R como 3 cores).
        frames: número de frames (se None, usa comprimento do histórico).
        Retorna o objeto FuncAnimation.
        """
        # map states to colors: 0->0, 1->1, 2->2
        if frames is None:
            frames = len(self.history['I'])
        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.imshow(self.grid, vmin=0, vmax=2, animated=True)
        ax.set_xticks([])
        ax.set_yticks([])
        title = ax.text(0.5, 1.01, f"t=0", transform=ax.transAxes, ha='center')

        # snapshot function: we will step the model and update the image
        def update(i):
            # se já não houver dados precomputados, fazer step
            if i < len(self.history['I'])-1:
                # se histórico já existe, reconstruir grid a partir de self.grid não é trivial
                pass
            # para simplicidade vamos executar steps a partir do próprio estado
            if i > 0:
                self.step()
            im.set_data(self.grid)
            title.set_text(f"t={self.t}")
            return (im, title)

        anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
        if savepath:
            anim.save(savepath, dpi=150)
        return anim