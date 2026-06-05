import numpy as np
import random
import gymnasium as gym
import math
from collections import defaultdict
import matplotlib.pyplot as graph

# ==========================================
# 1. Definisikan Masalah dan Environment
# ==========================================
# Tujuan: Melatih agen BipedalWalker-v3 untuk berjalan ke ujung rute 
# tanpa terjatuh menggunakan algoritma Q-Learning dengan teknik diskritisasi.
ENV_NAME = "BipedalWalker-v3"

# ==========================================
# 2. Formulasi MDP
# ==========================================
# State: 14 dimensi observasi kontinu yang didiskritisasi (dari total 24 state asli BipedalWalker)
# Action: 4 sendi (joints) kontinu yang didiskritisasi menjadi 10 nilai (-1 hingga 1)
# Reward: Didapatkan dari environment (maju = positif, jatuh = negatif, dll)
# Terminal Condition: terminated atau truncated (jatuh atau waktu habis)

stateBounds = [
    (0, math.pi), (-2,2), (-1,1), (-1,1),
    (0,math.pi), (-2,2), (0, math.pi), (-2,2),
    (0,1), (0, math.pi), (-2, 2), (0, math.pi),
    (-2, 2), (0, 1)
]

def discretizeState(state):
    discreteState = []
    for i in range(len(stateBounds)):
        val = state[i]
        low, high = stateBounds[i]
        # Batasi nilai agar tidak out of bounds
        val = max(low, min(val, high))
        # Skalakan ke index 0 hingga 9 (10 bin)
        index = int((val - low) / (high - low) * 9)
        index = min(9, index)
        discreteState.append(index)
    return tuple(discreteState)

def convertNextAction(nextAction):
    # Mengembalikan nilai diskrit (0-9) ke nilai kontinu environment (-1.0 hingga 1.0)
    action = []
    for val in nextAction:
        action.append((val / 9.0) * 2.0 - 1.0)
    return tuple(action)

# ==========================================
# 3. Arsitektur Model
# ==========================================
# Hyperparameter
EPISODES = 500  # Menggunakan 500 agar lebih realistis untuk ditunggu
GAMMA = 0.99
ALPHA = 0.1     # Diperbesar agar tabel Q lebih cepat update
EPSILON_DECAY = 0.004
HIGHSCORE = -1000
MAX_STEPS = 1000 # Batas langkah per episode agar agen tidak stuck menyeimbangkan diri

# Inisialisasi Q-Table
# Q-Table berbentuk dictionary dengan key = tuple State, 
# dan value = matriks aksi 4D ukuran (10, 10, 10, 10)
def create_q_table():
    return defaultdict(lambda: np.zeros((10, 10, 10, 10)))

# ==========================================
# 4. Training dan Eksplorasi
# ==========================================
def getNextAction(qTable, epsilon, state):
    # Epsilon-Greedy Exploration
    if random.random() < epsilon:
        # Explore: Pilih aksi random untuk ke-4 sendi (0-9)
        return tuple(random.randint(0, 9) for _ in range(4))
    else:
        # Exploit: Pilih aksi yang memaksimalkan nilai Q pada state tersebut
        return np.unravel_index(np.argmax(qTable[state]), qTable[state].shape)

def updateQTable(qTable, state, action, reward, nextState):
    current = qTable[state][action]  
    qNext = np.max(qTable[nextState])
    target = reward + (GAMMA * qNext)
    new_value = current + (ALPHA * (target - current))
    return new_value

def runAlgorithmStep(env, episode, qTable, doRender):
    global HIGHSCORE
    print(f"Episode #: {episode}", end=" | ")

    obs, _ = env.reset()
    state = discretizeState(obs[0:14])
    
    total_reward = 0
    # Epsilon menurun bertahap (Decay) seiring bertambahnya episode
    epsilon = max(0.01, 1.0 / (episode * EPSILON_DECAY))
    step_count = 0

    while True:
        step_count += 1
        if doRender:
            env.render()
            
        nextActionDiscretized = getNextAction(qTable, epsilon, state)
        nextActionContinuous = convertNextAction(nextActionDiscretized)
        
        # Interaksi dengan environment
        nextState_obs, reward, terminated, truncated, _ = env.step(nextActionContinuous)
        done = terminated or truncated
        
        # Observasi state baru & Update reward
        nextState = discretizeState(nextState_obs[0:14])
        total_reward += reward
        
        # Update model / Q-Table
        qTable[state][nextActionDiscretized] = updateQTable(qTable, state, nextActionDiscretized, reward, nextState)
        state = nextState
        
        if done or step_count >= MAX_STEPS:
            break
            
    print(f"Score: {total_reward:.2f} | Epsilon: {epsilon:.3f}")
    if total_reward > HIGHSCORE:
        HIGHSCORE = total_reward
        
    return total_reward

# ==========================================
# 5. Evaluasi
# ==========================================
def plotEpisode(myGraph, mySubPlot, xval, yval, plotLine, movingAvgLine, epScore, i):
    xval.append(i)
    yval.append(epScore)
    
    plotLine.set_xdata(xval)
    plotLine.set_ydata(yval)
    
    # Hitung Moving Average (berdasarkan 50 episode terakhir)
    window = 50
    if len(yval) >= window:
        moving_avg = np.convolve(yval, np.ones(window)/window, mode='valid')
        # Berikan padding None di awal agar sesuai ukuran x
        moving_avg_padded = [np.nan] * (window - 1) + list(moving_avg)
        movingAvgLine.set_xdata(xval)
        movingAvgLine.set_ydata(moving_avg_padded)
        
    # Plot tidak diperbarui ke layar secara real-time untuk kompatibilitas Notebook
    # Hasil akan disimpan dan ditampilkan di akhir program.

# ==========================================
# 6. Analisis Hasil & 7. Kesimpulan
# ==========================================
def main():
    global HIGHSCORE
    visualize = input("Visualize simulation? [y/n]\n").lower()
    doRender = (visualize == 'y')
    
    env = gym.make(ENV_NAME, hardcore=False, render_mode="human" if doRender else None)
    qTable = create_q_table()

    # --- Setup Evaluasi (Matplotlib) ---
    myGraph = graph.figure(figsize=(10, 5))
    mySubPlot = myGraph.add_subplot()
    graph.xlabel("Episode #")
    graph.ylabel("Score")
    graph.title("BipedalWalker Q-Learning: Scores vs Episode")
    
    xval, yval = [], []
    plotLine, = mySubPlot.plot(xval, yval, label='Reward per Episode', color='blue', alpha=0.5)
    movingAvgLine, = mySubPlot.plot([], [], label='Moving Average (50)', color='red', linewidth=2)
    mySubPlot.legend()

    print("\nMulai Training...")
    try:
        for i in range(1, EPISODES + 1):
            epScore = runAlgorithmStep(env, i, qTable, doRender)
            plotEpisode(myGraph, mySubPlot, xval, yval, plotLine, movingAvgLine, epScore, i)
    except KeyboardInterrupt:
        print("\nTraining dihentikan secara manual oleh pengguna (Ctrl+C).")

    # --- Menyimpan & Menampilkan Analisis Akhir ---
    mySubPlot.set_xlim([0, EPISODES])
    if len(yval) > 0:
        mySubPlot.set_ylim([min(-300.0, min(yval)), max(100.0, max(yval))])

    myGraph.savefig("./plot_hasil_training.png")
    print("\nGrafik diekspor sebagai plot_hasil_training.png")
    
    print("\n==========================================")
    print("6. Analisis Hasil")
    print("==========================================")
    print(f"-> Training selesai. Skor tertinggi yang pernah dicapai: {HIGHSCORE:.2f}")
    print("-> Karena Environment ini memiliki state kontinu yang sangat kompleks (14 state),")
    print("   Q-Table dengan diskritisasi dasar seringkali lambat atau gagal konvergen secara optimal.")
    
    print("\n==========================================")
    print("7. Kesimpulan")
    print("==========================================")
    print("-> Implementasi menunjukkan bagaimana memformulasikan masalah robot berjalan")
    print("   menjadi skema MDP (State-Action-Reward). Walau sederhana, agen dapat belajar")
    print("   seiring episodenya, ditunjukkan dengan perubahan moving average reward.")
    print("==========================================\n")

    graph.show() # Tampilkan plot terakhir secara statis sebelum program berhenti
    env.close()

if __name__ == "__main__":
    main()