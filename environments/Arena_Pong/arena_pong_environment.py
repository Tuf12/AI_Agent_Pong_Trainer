# arena_pong_environment.py - Pure Pong Game Logic (No Agent Byte Dependencies)
import numpy as np
import random
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class ArenaPongEnvironment:
    """
    Pure Arena Pong Environment - Game Logic Only

    This environment contains ONLY the Pong game logic and has no knowledge
    of Agent Byte, transfer learning, or adapters. It's a pure game that can
    be connected to any AI system through appropriate adapters.

    Features:
    - Center-zero coordinate system (Y=0 at center)
    - Multi-user arena support (spectating, tournaments)
    - Performance analytics and metrics tracking
    - User demonstration recording
    - Pure game state management
    """

    def __init__(self, width=800, height=400, match_id: Optional[str] = None, is_arena_match: bool = False):
        print("🏓 Initializing Pure Arena Pong Environment...")

        # Game settings
        self.width = width
        self.height = height
        self.paddle_height = 80
        self.paddle_width = 10
        self.ball_size = 10
        self.paddle_speed = 12
        self.ball_speed = 12
        self.default_ball_speed = 12
        self.winning_score = 21

        # Arena-specific settings
        self.match_id = match_id or f"match_{int(time.time())}"
        self.is_arena_match = is_arena_match
        self.spectators = set()
        self.match_start_time = None
        self.match_duration = 0.0
        
        # State normalization settings (for adapter compatibility)
        self.raw_state_size = 14  # Original Pong state dimensions
        self.standardized_state_size = 256  # Target for transfer learning

        # Game state
        self.running = False
        self.game_over = False
        self.winner = None
        self.match_result = None

        # Positions (center-zero coordinate system: Y=0 at center)
        self.ball_x = 0
        self.ball_y = 0
        self.ball_dx = 0
        self.ball_dy = 0
        self.player_paddle_y = 0  # Y=0 is center of screen
        self.ai_paddle_y = 0  # Y=0 is center of screen
        self.player_score = 0
        self.ai_score = 0

        # Performance tracking (pure game metrics)
        self.previous_ball_x = 0
        self.previous_ball_y = 0
        self.last_ai_action = 0
        self.action_history = []
        self.state_history = []

        # Game-specific task tracking
        self.task_successes = 0  # Ball interceptions
        self.task_failures = 0  # Ball misses
        self.success_streak = 0
        self.best_success_streak = 0
        self.total_success_bonus = 0.0
        self.total_failure_penalty = 0.0
        self.task_completions = 0  # Goals scored
        self.task_failures_major = 0  # Goals conceded

        # Performance analytics (pure game data)
        self.performance_metrics = {
            'reaction_times': [],
            'prediction_accuracy': [],
            'positioning_effectiveness': [],
            'adaptation_events': [],
            'strategic_decisions': []
        }

        # User demonstration tracking
        self.user_action_history = []
        self.last_user_action = None
        self.last_user_state = None
        self.user_demonstration_active = False
        self.user_successful_patterns = []

        self.reset_game()

        print("✅ Pure Arena Pong Environment Ready!")
        print(f"   🎮 Match ID: {self.match_id}")
        print(f"   🏟️ Arena Mode: {self.is_arena_match}")
        print(f"   🎯 Pure Game Logic - No AI Dependencies")
        print(f"   📊 Ready for Adapter Integration")

    # ========================================
    # CORE GAME LOGIC (PURE PONG)
    # ========================================

    def reset_game(self):
        """Reset game state for new match"""
        print(f"🔄 Resetting Pure Pong Environment (Match: {self.match_id})")

        # Reset positions to center-zero system
        self.ball_x = 0
        self.ball_y = 0
        self.ball_dx = random.choice([-1, 1]) * self.ball_speed
        self.ball_dy = random.uniform(-2, 2)

        self.player_paddle_y = 0  # Center of screen
        self.ai_paddle_y = 0  # Center of screen
        self.player_score = 0
        self.ai_score = 0

        # Reset tracking
        self.previous_ball_x = self.ball_x
        self.previous_ball_y = self.ball_y
        self.last_ai_action = 0

        # Reset task tracking
        self.task_successes = 0
        self.task_failures = 0
        self.success_streak = 0
        self.total_success_bonus = 0.0
        self.total_failure_penalty = 0.0
        self.task_completions = 0
        self.task_failures_major = 0

        # Reset analytics
        self.action_history = []
        self.state_history = []
        self.performance_metrics = {key: [] for key in self.performance_metrics.keys()}

        # Reset game state
        self.game_over = False
        self.winner = None
        self.match_result = None
        self.running = False
        self.match_start_time = None

        print("✅ Arena Pong environment reset complete")

    def start_match(self):
        """Start arena match with timing"""
        self.running = True
        self.match_start_time = time.time()
        print(f"🚀 Arena match started: {self.match_id}")

    def create_state(self) -> np.ndarray:
        """Create raw 14-dimension Pong state"""
        # Center-zero coordinate system normalization
        state = np.array([
            self.ball_x / (self.width / 2),  # Normalized ball position
            self.ball_y / (self.height / 2),
            self.ball_dx / self.ball_speed,  # Normalized ball velocity
            self.ball_dy / self.ball_speed,
            self.player_paddle_y / (self.height / 2),  # Normalized paddle positions
            self.ai_paddle_y / (self.height / 2),
            (self.player_score - self.ai_score) / 21,  # Score difference
            self.success_streak / 10,  # Normalized success streak
            self.task_successes / max(1, self.task_successes + self.task_failures),  # Success rate
            self.ball_x * self.ball_dx,  # Ball momentum toward AI
            abs(self.ball_y - self.ai_paddle_y) / (self.height / 2),  # Ball-paddle distance
            1.0 if self.ball_dx > 0 else -1.0,  # Ball direction
            min(1.0, self.success_streak / 5),  # Streak bonus factor
            1.0 if self.running else 0.0  # Game active flag
        ], dtype=np.float32)

        return state

    def normalize_state_to_256d(self, raw_state: np.ndarray) -> np.ndarray:
        """
        DEPRECATED: This should be handled by adapters, not the environment

        Pure game environments should not know about AI systems.
        Use PongArenaAdapter.normalize_state_to_256d() instead.
        """
        raise NotImplementedError("Use PongArenaAdapter for state normalization")

    def get_env_context(self) -> Dict[str, Any]:
        """
        DEPRECATED: This should be handled by adapters, not the environment

        Pure game environments should not provide transfer learning context.
        Use PongArenaAdapter.get_transfer_context() instead.
        """
        raise NotImplementedError("Use PongArenaAdapter for transfer learning context")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute one step in the environment with enhanced analytics"""
        if not self.running or self.game_over:
            return self.create_state(), 0.0, True

        # Record action for analytics
        action_record = {
            'timestamp': time.time(),
            'action': action,
            'state_before': self.create_state().copy(),
            'ball_position': [self.ball_x, self.ball_y],
            'paddle_position': self.ai_paddle_y
        }

        # Store previous state for reward calculation
        prev_state = self.create_state()
        self.previous_ball_x = self.ball_x
        self.previous_ball_y = self.ball_y
        self.last_ai_action = action

        # Execute AI action
        if action == 0:  # Move up
            self.ai_paddle_y -= self.paddle_speed
        elif action == 2:  # Move down
            self.ai_paddle_y += self.paddle_speed
        # action == 1 means stay

        # Keep paddle in bounds (center-zero system)
        self.ai_paddle_y = max(-self.height / 2 + self.paddle_height / 2,
                               min(self.height / 2 - self.paddle_height / 2, self.ai_paddle_y))

        # Update ball position
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # Ball collision with top/bottom walls
        if self.ball_y <= -self.height / 2 + self.ball_size / 2 or self.ball_y >= self.height / 2 - self.ball_size / 2:
            self.ball_dy = -self.ball_dy

        # Initialize reward
        reward = 0.0
        hit_occurred = False

        # AI paddle collision (right side)
        if (self.ball_x >= self.width / 2 - 30 and
                self.ball_x <= self.width / 2 - 10 and
                self.ball_dx > 0 and
                self.ball_y >= self.ai_paddle_y - self.paddle_height / 2 and
                self.ball_y <= self.ai_paddle_y + self.paddle_height / 2):

            hit_occurred = True
            self.ball_dx = -abs(self.ball_dx)

            # Add spin based on paddle position
            paddle_center = self.ai_paddle_y
            hit_pos = (self.ball_y - paddle_center) / (self.paddle_height / 2)
            self.ball_dy += hit_pos * 2

            # Symbolic reward: Task Success
            self.task_successes += 1
            self.success_streak += 1
            self.best_success_streak = max(self.best_success_streak, self.success_streak)

            streak_bonus = 0.25 * min(self.success_streak, 10)
            reward = 1.0 + streak_bonus
            self.total_success_bonus += reward

            # Track performance metrics
            self.performance_metrics['reaction_times'].append(time.time())

            # Calculate prediction accuracy
            expected_y = self.ball_y
            actual_paddle_y = self.ai_paddle_y
            prediction_error = abs(expected_y - actual_paddle_y)
            prediction_accuracy = max(0, 1 - prediction_error / (self.paddle_height / 2))
            self.performance_metrics['prediction_accuracy'].append(prediction_accuracy)

            print(f"🎯 AI Hit! Streak: {self.success_streak}, Reward: {reward:.2f}")

        # Player paddle collision (left side)
        elif (self.ball_x <= -self.width / 2 + 30 and
              self.ball_x >= -self.width / 2 + 10 and
              self.ball_dx < 0 and
              self.ball_y >= self.player_paddle_y - self.paddle_height / 2 and
              self.ball_y <= self.player_paddle_y + self.paddle_height / 2):

            self.ball_dx = abs(self.ball_dx)
            paddle_center = self.player_paddle_y
            hit_pos = (self.ball_y - paddle_center) / (self.paddle_height / 2)
            self.ball_dy += hit_pos * 2

        # Scoring (AI scores - right side)
        elif self.ball_x <= -self.width / 2:
            self.ai_score += 1
            self.task_completions += 1
            reward = 3.0  # Task Completion reward
            print(f"🎯 AI SCORES! Score: {self.player_score}-{self.ai_score}")
            self._reset_ball()

        # Player scores (left side)
        elif self.ball_x >= self.width / 2:
            self.player_score += 1
            self.task_failures_major += 1
            self.success_streak = 0  # Reset streak
            reward = -0.5  # Minor penalty
            print(f"😔 Player scores. Score: {self.player_score}-{self.ai_score}")
            self._reset_ball()

        # Miss penalty (ball passed AI without hitting)
        elif (self.ball_x > self.width / 2 - 50 and
              self.previous_ball_x <= self.width / 2 - 50 and
              self.ball_dx > 0 and not hit_occurred):

            self.task_failures += 1
            self.success_streak = 0
            reward = -0.5  # Task Failure penalty
            self.total_failure_penalty += abs(reward)
            print(f"❌ AI Miss! Failures: {self.task_failures}")

        # Small positioning reward
        ball_paddle_distance = abs(self.ball_y - self.ai_paddle_y)
        if ball_paddle_distance < self.paddle_height:
            reward += 0.1 * (1 - ball_paddle_distance / self.paddle_height)

        # Update action record with results
        action_record.update({
            'reward': reward,
            'success': hit_occurred,
            'state_after': self.create_state().copy(),
            'performance_change': reward > 0
        })
        self.action_history.append(action_record)

        # Keep history manageable
        if len(self.action_history) > 1000:
            self.action_history = self.action_history[-500:]

        # Store state history for pattern recognition
        current_state = self.create_state()
        self.state_history.append(current_state)
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-50:]

        # Check win condition
        if self.ai_score >= self.winning_score:
            self.game_over = True
            self.winner = "Agent Byte"
            reward += 10.0  # Match Win bonus
            self.match_result = "ai_win"
            print(f"🏆 AI WINS! Final Score: {self.player_score}-{self.ai_score}")
        elif self.player_score >= self.winning_score:
            self.game_over = True
            self.winner = "Player"
            reward -= 10.0  # Match Loss penalty
            self.match_result = "player_win"
            print(f"😞 Player wins. Final Score: {self.player_score}-{self.ai_score}")

        # Calculate match duration
        if self.match_start_time:
            self.match_duration = time.time() - self.match_start_time

        return current_state, reward, self.game_over

    def _reset_ball(self):
        """Reset ball to center with random direction"""
        self.ball_x = 0
        self.ball_y = random.uniform(-50, 50)
        self.ball_dx = random.choice([-1, 1]) * self.ball_speed
        self.ball_dy = random.uniform(-2, 2)

    def move_player_paddle(self, direction: int):
        """Move player paddle and record demonstration data"""
        self.last_user_action = direction
        self.last_user_state = self.create_state().copy()

        if direction == -1:  # Up
            self.player_paddle_y -= self.paddle_speed
        elif direction == 1:  # Down
            self.player_paddle_y += self.paddle_speed

        # Keep in bounds (center-zero system)
        self.player_paddle_y = max(-self.height / 2 + self.paddle_height / 2,
                                   min(self.height / 2 - self.paddle_height / 2, self.player_paddle_y))

        # Record user action for learning
        user_action_record = {
            'timestamp': time.time(),
            'action': direction,
            'state': self.last_user_state,
            'paddle_position': self.player_paddle_y,
            'ball_position': [self.ball_x, self.ball_y],
            'ball_velocity': [self.ball_dx, self.ball_dy]
        }
        self.user_action_history.append(user_action_record)

        # Keep reasonable history size
        if len(self.user_action_history) > 200:
            self.user_action_history = self.user_action_history[-100:]

    def evaluate_user_action_outcome(self) -> Optional[Dict[str, Any]]:
        """Evaluate the outcome of user actions for learning"""
        if not self.user_action_history:
            return None

        recent_action = self.user_action_history[-1]

        # Simple evaluation based on ball-paddle distance improvement
        ball_paddle_distance = abs(self.ball_y - self.player_paddle_y)

        evaluation = {
            'action': recent_action['action'],
            'state': recent_action['state'],
            'outcome_quality': 1.0 if ball_paddle_distance < self.paddle_height / 2 else 0.0,
            'distance_to_ball': ball_paddle_distance,
            'timing_quality': 1.0 if self.ball_dx < 0 else 0.5,  # Better if ball approaching
            'timestamp': time.time()
        }

        return evaluation

    def get_game_state(self) -> Dict[str, Any]:
        """Get comprehensive game state for UI and analytics"""
        return {
            'match_id': self.match_id,
            'running': self.running,
            'game_over': self.game_over,
            'winner': self.winner,
            'match_result': self.match_result,
            'match_duration': self.match_duration,

            # Positions (converted to screen coordinates for UI)
            'ball': {
                'x': self.ball_x + self.width / 2,
                'y': self.ball_y + self.height / 2,
                'size': self.ball_size
            },
            'player_paddle': {
                'y': self.player_paddle_y + self.height / 2 - self.paddle_height / 2,
                'height': self.paddle_height
            },
            'ai_paddle': {
                'y': self.ai_paddle_y + self.height / 2 - self.paddle_height / 2,
                'height': self.paddle_height
            },
            'scores': {
                'player': self.player_score,
                'ai': self.ai_score,
                'winning_score': self.winning_score
            },
            'dimensions': {
                'width': self.width,
                'height': self.height
            },

            # Performance metrics
            'task_successes': self.task_successes,
            'task_failures': self.task_failures,
            'success_streak': self.success_streak,
            'best_success_streak': self.best_success_streak,
            'task_completions': self.task_completions,

            # Arena features
            'is_arena_match': self.is_arena_match,
            'spectator_count': len(self.spectators),

            # Transfer learning context
            'transfer_learning_enabled': True,
            'standardized_state_size': self.standardized_state_size,
            'environment_type': 'arena_pong'
        }

    def get_pong_stats(self) -> Dict[str, Any]:
        """Get detailed Pong-specific statistics"""
        total_attempts = self.task_successes + self.task_failures
        hit_rate = (self.task_successes / max(1, total_attempts)) * 100

        avg_prediction_accuracy = 0.0
        if self.performance_metrics['prediction_accuracy']:
            avg_prediction_accuracy = np.mean(self.performance_metrics['prediction_accuracy'])

        return {
            'hit_rate': hit_rate,
            'total_hits': self.task_successes,
            'total_misses': self.task_failures,
            'best_streak': self.best_success_streak,
            'current_streak': self.success_streak,
            'goals_scored': self.task_completions,
            'goals_conceded': self.task_failures_major,
            'total_reward_earned': self.total_success_bonus - self.total_failure_penalty,
            'average_prediction_accuracy': avg_prediction_accuracy,
            'total_actions': len(self.action_history),
            'match_duration': self.match_duration,
            'actions_per_second': len(self.action_history) / max(1,
                                                                 self.match_duration) if self.match_duration > 0 else 0
        }

    # ========================================
    # ARENA FEATURES (PURE GAME LOGIC)
    # ========================================

    def add_spectator(self, spectator_id: str):
        """Add spectator to arena match"""
        self.spectators.add(spectator_id)
        print(f"👥 Spectator {spectator_id} joined match {self.match_id}")

    def remove_spectator(self, spectator_id: str):
        """Remove spectator from arena match"""
        self.spectators.discard(spectator_id)
        print(f"👥 Spectator {spectator_id} left match {self.match_id}")

    def get_transfer_learning_metrics(self) -> Dict[str, Any]:
        """Get metrics specifically for transfer learning analysis"""
        total_attempts = self.task_successes + self.task_failures

        return {
            'environment_experience': {
                'total_actions': len(self.action_history),
                'successful_actions': self.task_successes,
                'completion_events': self.task_completions,
                'adaptation_events': len(self.performance_metrics['adaptation_events'])
            },
            'skill_development': {
                'prediction_skill': np.mean(self.performance_metrics['prediction_accuracy']) if
                self.performance_metrics['prediction_accuracy'] else 0,
                'positioning_skill': self.best_success_streak / 20.0,
                'timing_skill': min(1.0, self.task_successes / max(1,
                                                                   self.task_failures)) if self.task_failures > 0 else 1.0,
                'adaptation_skill': len(
                    set(a['action'] for a in self.action_history[-20:])) / 3.0 if self.action_history else 0
            },
            'transferable_patterns': {
                'successful_trajectories': [a for a in self.action_history if a.get('success', False)][-10:],
                'error_recovery_patterns': [],  # Could be enhanced
                'strategic_adaptations': []  # Could be enhanced
            },
            'readiness_indicators': {
                'experience_level': min(1.0, len(self.action_history) / 100),
                'consistency_score': min(1.0, self.best_success_streak / 10),
                'skill_diversity': len(
                    set(a['action'] for a in self.action_history)) / 3.0 if self.action_history else 0,
                'transfer_readiness': min(1.0, (self.task_successes / max(1, total_attempts)) * (
                        len(self.action_history) / 50)) if total_attempts > 0 else 0.0
            }
        }

    def get_env_context(self) -> Dict[str, Any]:
        """
        🌍 Provide comprehensive Arena Pong context for agent understanding
        
        This is what teaches the agent what Arena Pong IS and how to succeed
        """
        return {
            'environment_id': 'arena_pong',
            'environment_type': 'competitive_real_time',
            'display_name': 'Arena Pong',
            
            # 🎯 PRIMARY OBJECTIVE - What should the agent do?
            'objective': {
                'primary': 'Score 21 points before opponent by hitting ball with paddle',
                'secondary': [
                    'Keep ball in play as long as possible',
                    'Prevent opponent from scoring',
                    'Maintain consistent paddle control'
                ],
                'victory_conditions': [
                    'First player to reach 21 points wins',
                    'Opponent disconnects or forfeits'
                ],
                'failure_conditions': [
                    'Opponent reaches 21 points first',
                    'Connection lost or game timeout'
                ]
            },
            
            # 📋 GAME RULES - How does Arena Pong work?
            'rules': [
                'Hit ball with paddle to keep it in play',
                'Ball bounces off top and bottom walls',
                'Score 1 point when ball passes opponent paddle',
                'Ball resets to center after each point',
                'Paddle can only move up and down',
                'Paddle must stay within screen boundaries',
                'First to 21 points wins the match'
            ],
            
            # 🎮 GAME MECHANICS - Technical details
            'game_mechanics': {
                'scoring': [
                    'Ball passing left edge = AI scores (right paddle wins)',
                    'Ball passing right edge = Player scores (left paddle wins)',
                    'Points are cumulative until match end'
                ],
                'paddle_control': [
                    'Action 0 = Move paddle up',
                    'Action 1 = Keep paddle stationary', 
                    'Action 2 = Move paddle down'
                ],
                'ball_physics': [
                    'Ball bounces off top/bottom walls',
                    'Ball speed may increase during rally',
                    'Paddle contact changes ball direction',
                    'Ball angle depends on paddle hit position'
                ],
                'constraints': [
                    'Cannot move paddle outside screen bounds',
                    'Cannot directly control ball movement',
                    'Must react to ball physics in real-time'
                ]
            },
            
            # 🧠 STRATEGIC CONCEPTS - How to succeed?
            'strategic_concepts': {
                'core_skills': [
                    'Ball trajectory prediction and interception',
                    'Optimal paddle positioning for defense and offense',
                    'Timing optimization for precise ball contact',
                    'Opponent movement pattern recognition',
                    'Error recovery after missed hits',
                    'Adaptive strategy based on game state'
                ],
                'tactical_approaches': [
                    'Predictive positioning - move to where ball will be',
                    'Defensive positioning - stay centered when ball moving away',
                    'Aggressive positioning - force difficult returns',
                    'Pattern-based prediction - learn opponent tendencies',
                    'Momentum control - manage ball speed and angle'
                ],
                'success_patterns': [
                    'Early positioning beats reactive movement',
                    'Consistent prediction outperforms random actions',
                    'Adaptive timing based on ball velocity changes',
                    'Center positioning provides maximum coverage',
                    'Calm recovery after errors maintains performance'
                ],
                'failure_patterns': [
                    'Purely reactive movement leads to missed hits',
                    'Over-aggressive positioning leaves gaps',
                    'Inconsistent timing creates unpredictable results',
                    'Ignoring opponent patterns misses opportunities',
                    'Poor error recovery leads to losing streaks'
                ]
            },
            
            # 🔄 TRANSFERABLE SKILLS - What can be learned here?
            'transferable_skills': [
                'trajectory_prediction',     # Predict moving object paths
                'timing_optimization',       # Execute actions at optimal moments  
                'strategic_positioning',     # Position optimally for advantage
                'pattern_recognition',       # Recognize and adapt to patterns
                'error_recovery',           # Recover from mistakes effectively
                'adaptive_strategy',        # Modify strategy based on conditions
                'competitive_decision_making', # Make decisions under pressure
                'real_time_responsiveness'  # React quickly to changing situations
            ],
            
            # 🎓 LEARNING RECOMMENDATIONS - How should the agent focus?
            'learning_recommendations': {
                'neural_focus': [
                    'Ball position and velocity pattern recognition',
                    'Paddle movement optimization for interception',
                    'Reward correlation with positioning accuracy',
                    'Temporal sequence learning for prediction'
                ],
                'symbolic_focus': [
                    'Strategic positioning rules and principles',
                    'Opponent behavior pattern analysis',
                    'Error recovery strategy development',
                    'Win condition optimization approaches'
                ],
                'transfer_focus': [
                    'Universal prediction and interception skills',
                    'Competitive strategy and adaptation patterns',
                    'Real-time decision making under pressure',
                    'Pattern recognition across dynamic environments'
                ]
            },
            
            # 📊 PERFORMANCE METRICS - How to measure success?
            'performance_metrics': {
                'primary_metrics': [
                    'Win rate (matches won / total matches)',
                    'Average points scored per match',
                    'Hit rate (successful ball contacts / opportunities)'
                ],
                'skill_development_metrics': [
                    'Ball prediction accuracy over time',
                    'Positioning effectiveness score',
                    'Error recovery speed after misses',
                    'Adaptation rate to opponent changes'
                ],
                'transfer_learning_metrics': [
                    'Cross-session skill retention',
                    'Strategy abstraction capability', 
                    'Pattern recognition generalization',
                    'Decision making consistency'
                ]
            },
            
            # 🏆 SUCCESS INDICATORS - What shows the agent is learning?
            'success_indicators': {
                'beginner_level': [
                    'Can hit ball consistently (>50% hit rate)',
                    'Understands basic paddle movement',
                    'Recognizes scoring opportunities'
                ],
                'intermediate_level': [
                    'Demonstrates predictive positioning',
                    'Shows pattern recognition in play',
                    'Maintains consistent performance'
                ],
                'advanced_level': [
                    'Exhibits strategic adaptation',
                    'Shows opponent pattern learning',
                    'Demonstrates error recovery skills'
                ],
                'expert_level': [
                    'Consistently wins against varied opponents',
                    'Shows transferable skill development',
                    'Demonstrates meta-learning capabilities'
                ]
            },
            
            # 🎯 ARENA-SPECIFIC FEATURES
            'arena_features': {
                'match_format': 'Best of 1 to 21 points',
                'real_time_gameplay': True,
                'spectator_support': True,
                'multi_agent_battles': True,
                'performance_analytics': True,
                'skill_transfer_tracking': True
            }
        }


# Test Arena Pong Context Integration
def test_arena_pong_context():
    """Test that Arena Pong properly provides context to agents"""
    
    print("🧪 Testing Arena Pong Context Integration...")
    
    # Create Arena Pong environment
    env = ArenaPongEnvironment(match_id="test_context", is_arena_match=True)
    
    # Get context
    context = env.get_env_context()
    
    print(f"✅ Context Retrieved:")
    print(f"   Environment: {context['environment_id']}")
    print(f"   Primary Objective: {context['objective']['primary']}")
    print(f"   Rules Count: {len(context['rules'])}")
    print(f"   Strategic Skills: {len(context['strategic_concepts']['core_skills'])}")
    print(f"   Transferable Skills: {len(context['transferable_skills'])}")
    
    print(f"🚀 Arena Pong Context Integration: COMPLETE!")
    return True


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Arena Pong Environment v2.0...")

    # Test context integration
    test_arena_pong_context()

    print("✅ All tests completed successfully!")