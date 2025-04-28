import chess
import numpy as np


class ChessEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.board = chess.Board()
        self.action_space = gym.spaces.Discrete(4672)  # 4672 possible moves in chess - AlphaZero
        self.observation_space = gym.spaces.Box(low=-6, high=6, shape=(8,8), dtype=np.float32)
        self._piece_map_to_index = {
            chess.Piece.from_symbol('P'): 1,
            chess.Piece.from_symbol('N'): 2,
            chess.Piece.from_symbol('B'): 3,
            chess.Piece.from_symbol('R'): 4,
            chess.Piece.from_symbol('Q'): 5,
            chess.Piece.from_symbol('K'): 6,
            chess.Piece.from_symbol('p'): -1,
            chess.Piece.from_symbol('n'): -2,
            chess.Piece.from_symbol('b'): -3,
            chess.Piece.from_symbol('r'): -4,
            chess.Piece.from_symbol('q'): -5,
            chess.Piece.from_symbol('k'): -6,
        }
        
        self.step_count = 0
        self.episode = 0
        
        self.render_mode = render_mode
        
        with open ("./logs/actions.csv", "w") as f:
            f.write("episode,step,move,reward\n")
            
        
    def _get_obs(self):
        piece_map = self.board.piece_map()
        obs = np.zeros((8, 8,), dtype=np.int8)
        for square, piece in piece_map.items():
            obs[chess.square_rank(square), chess.square_file(square)] = self._piece_map_to_index[piece]
        
        return obs.astype(np.float32)
       
    
    def _get_info(self):
        return {
            'legal_moves': [self._piece_map_to_index[piece] for square, piece in self.board.piece_map().items()],
            'is_check': self.board.is_check(),
            'is_checkmate': self.board.is_checkmate(),
            'is_stalemate': self.board.is_stalemate(),
            'is_insufficient_material': self.board.is_insufficient_material(),
            'is_seventyfive_moves': self.board.is_seventyfive_moves(),
            'is_fivefold_repetition': self.board.is_fivefold_repetition(),
        }
    
    def _get_action_dict(self):
        action_to_move_dict = {}
        for i, move in enumerate(self.board.legal_moves):
            action_to_move_dict[i] = move
        return action_to_move_dict    
    
    def action_to_move(self, action):
        action_to_move_dict = self._get_action_dict()
        if action in action_to_move_dict:
            return action_to_move_dict[action]
        else:
            # if the chosen action is illegal, select a random legal move
            action = np.random.choice(list(action_to_move_dict.keys()))
            return action_to_move_dict[action]
    
    def step(self, action):
        legal_moves_dict = self._get_action_dict()
                
        legal_moves_mask = np.zeros(self.action_space.n, dtype=np.int8)
        
        for idx in legal_moves_dict.keys():
            legal_moves_mask[idx] = 1
            
        move = self.action_to_move(action)
        reward = 0.1
        
        if self.step_count > 1:
            previous_move = self.board.peek()
            destination_square = previous_move.to_square
            if self.board.is_capture(previous_move):
                lost_piece = self.board.piece_at(destination_square)
                reward -= 0.04 * np.abs(self._piece_map_to_index[lost_piece])
        
        if self.board.is_capture(move):
            captured_piece = self.board.piece_at(move.to_square)
            if captured_piece:
                reward += 0.05 * np.abs(self._piece_map_to_index[captured_piece])
                
        if self.board.gives_check(move):
            reward += 0.5
        
        if self.board.is_checkmate() or self.board.is_stalemate():
            reward = -1
        elif self.board.is_check():
            reward = -0.95
    
        self.board.push(move)
        
        if self.board.is_checkmate():
            reward = 1
        elif self.board.is_stalemate():
            reward = -1
        
        if self.render_mode == 'human':
            self.render()
            
        self.step_count += 1
        with open ("./logs/actions.csv", "a") as f:
            f.write(f"{self.episode},{self.step_count},{move},{reward}\n")
        return self._get_obs(), reward, self.board.is_game_over(), False, {**self._get_info(), 'legal_moves_mask': legal_moves_mask}

    
    def reset(self):
        self.board.reset()
        self.step_count = 0
        self.episode += 1
        return self._get_obs(), self._get_info()
    
    def render(self):
        print("".join(["-" for _ in range (15)]))
        print(f"Current turn: {self.board.turn}")
        print(self.board)
        print("".join(["-" for _ in range (15)]))
        
def make_chess_env():
    return ChessEnv(render_mode='human')