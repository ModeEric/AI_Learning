#For project in future, held here for now
import os
import chess
import chess.engine
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, set_seed
from datasets import Dataset
from tqdm import tqdm

# Set random seed for reproducibility
set_seed(42)

# Load the model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Change to a smaller model if necessary
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load the model with a value head for PPO
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, config=config)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set up Stockfish engine for move validation and evaluation
engine_path = "/usr/games/stockfish"  # Change this to your Stockfish path
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

# Custom tokens
THINK_TOKEN = "[THINK]"
MOVE_TOKEN = "[MOVE]"
MAX_DEPTH = 1  # Depth for Stockfish analysis to evaluate move quality

# PPO Configuration
ppo_config = PPOConfig(
    batch_size=1,           # Because each game is a sequence
    forward_batch_size=1,   # Forward pass batch size
    log_with=None,          # Set to 'wandb' to log with Weights & Biases
    learning_rate=1.41e-5,
    adap_kl_ctrl=False,
    ppo_epochs=4,
)

# Initialize PPO Trainer
ppo_trainer = PPOTrainer(
    model,
    tokenizer,
    **ppo_config.__dict__,
)

# ELO Estimation Variables
MODEL_ELO = 800  # Starting ELO for the model
K_FACTOR = 32    # K-factor for ELO rating calculation

# Curriculum Learning Variables
STOCKFISH_LEVELS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]  # Stockfish skill levels (0-20)
current_level_index = 0  # Start with the easiest level

def set_stockfish_skill_level(level):
    # Set Stockfish skill level (0 to 20)
    engine.configure({"Skill Level": level})

# Set initial Stockfish skill level
set_stockfish_skill_level(STOCKFISH_LEVELS[current_level_index])

def generate_reasoning_and_move(query, model, tokenizer, max_length=200, temperature=0.7):
    # Encode the input prompt
    inputs = tokenizer(query, return_tensors="pt").to(device)
    
    # Generate reasoning and move
    output = model.generate(
        **inputs,
        max_length=inputs.input_ids.shape[1] + max_length,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Decode the output
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_text

def get_reward(board, move_san):
    try:
        # Parse the move
        move = board.parse_san(move_san)
        
        # Check if the move is legal
        if not board.is_legal(move):
            return INVALID_MOVE_PENALTY
        
        # Play the move on the board
        board.push(move)
        
        # Evaluate the new board state
        info = engine.analyse(board, chess.engine.Limit(depth=MAX_DEPTH))
        score = info["score"].relative.score(mate_score=10000)
        
        # Undo the move
        board.pop()
        
        # Normalize the score to a reasonable reward
        if score is None:
            return 0
        return score / 100  # Convert centipawns to pawns
    
    except Exception as e:
        # Invalid move
        return INVALID_MOVE_PENALTY

def play_game():
    global MODEL_ELO, current_level_index
    
    # Initialize chess board
    board = chess.Board()
    game_queries = []
    game_responses = []
    game_rewards = []
    
    # Track game result: 1=win, 0.5=draw, 0=loss
    game_result = None
    
    while not board.is_game_over():
        # Prepare the prompt
        prompt = f"Given the following board state:\n{board}\nWhat should be my next move? {THINK_TOKEN} "
        query = prompt
        
        # Generate reasoning and move
        response = generate_reasoning_and_move(query, model, tokenizer)
        # Extract move after MOVE_TOKEN
        if MOVE_TOKEN in response:
            reasoning, move = response.split(MOVE_TOKEN)
        else:
            reasoning = response
            move = ""
        
        move = move.strip()
        
        # Append query and response
        game_queries.append(query)
        game_responses.append(response)
        
        # Validate move and calculate reward
        reward = get_reward(board, move)
        game_rewards.append(torch.tensor(reward, device=device, dtype=torch.float32))
        
        # If the move is invalid, end the game
        if reward == INVALID_MOVE_PENALTY:
            game_result = 0  # Loss due to invalid move
            break
        else:
            # Play the move
            board.push_san(move)
        
        # Opponent move (Stockfish)
        result = engine.play(board, chess.engine.Limit(time=0.1))
        if result.move is None:
            # Stockfish resigns or cannot move
            game_result = 1  # Model wins
            break
        board.push(result.move)
    
    # Determine game outcome if not already set
    if game_result is None:
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                game_result = 1  # Model wins (black's turn and checkmate)
            else:
                game_result = 0  # Model loses
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            game_result = 0.5  # Draw
        else:
            game_result = 0  # Model loses
    
    # Update ELO rating
    opponent_elo = get_stockfish_elo(STOCKFISH_LEVELS[current_level_index])
    MODEL_ELO = update_elo(MODEL_ELO, opponent_elo, game_result)
    
    # Adjust Stockfish level based on model's ELO (Curriculum Learning)
    adjust_stockfish_level()
    
    return game_queries, game_responses, game_rewards, game_result

def get_stockfish_elo(level):
    # Approximate Stockfish ELO ratings for different skill levels
    # These are rough estimates; adjust as needed
    base_elo = 1300  # Approximate ELO at skill level 1
    elo_increase_per_level = 50
    return base_elo + (level - 1) * elo_increase_per_level

def update_elo(player_elo, opponent_elo, score):
    # Calculate expected score
    expected_score = 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))
    # Update ELO rating
    new_elo = player_elo + K_FACTOR * (score - expected_score)
    return new_elo

def adjust_stockfish_level():
    global current_level_index, STOCKFISH_LEVELS
    # Increase Stockfish level if model's ELO surpasses certain thresholds
    # Define thresholds corresponding to ELO ratings
    thresholds = [1000, 1200, 1400, 1600, 1800, 2000]
    
    # Check if we can increase the difficulty
    for i, threshold in enumerate(thresholds):
        if MODEL_ELO >= threshold and current_level_index < len(STOCKFISH_LEVELS) - 1:
            current_level_index = i + 1
            set_stockfish_skill_level(STOCKFISH_LEVELS[current_level_index])
            print(f"Model ELO: {MODEL_ELO:.1f}. Increasing Stockfish level to {STOCKFISH_LEVELS[current_level_index]}")
            break

# Define constants
INVALID_MOVE_PENALTY = -10
NUM_GAMES = 50  # Adjust based on your compute capability

# Training loop
for game_num in range(NUM_GAMES):
    print(f"\nStarting game {game_num + 1} against Stockfish level {STOCKFISH_LEVELS[current_level_index]}...")
    queries, responses, rewards, game_result = play_game()
    
    # Prepare data for PPO
    batch = {
        "query": queries,
        "response": responses,
        "reward": rewards,
    }
    
    # Run PPO step
    stats = ppo_trainer.step(batch["query"], batch["response"], batch["reward"])
    print(f"Game {game_num + 1} completed. PPO step done.")
    print(f"Game result: {'Win' if game_result == 1 else 'Draw' if game_result == 0.5 else 'Loss'}")
    print(f"Updated Model ELO: {MODEL_ELO:.1f}")
    
# Save the trained model
output_dir = "./trained_chess_model"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Close Stockfish engine
engine.quit()
