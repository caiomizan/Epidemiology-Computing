# Simulação Espacial SIR com Vacinação

Este projeto implementa uma simulação epidemiológica baseada no modelo **SIR** (Suscetível–Infectado–Recuperado), estendida para incluir um quarto estado **Vacinado (V)**, em uma grade bidimensional (autômato celular).  
A simulação permite explorar:
- Transmissão local por vizinhança (Moore – 8 vizinhos)
- Recuperação estocástica ou determinística
- Mobilidade de indivíduos

## Dependências
- Python 3.12+
- [NumPy]
- [Matplotlib]

Instale as dependências executando:
```bash
pip install -r requirements.txt
```
## Para Executar
```bash
python main.py
```
