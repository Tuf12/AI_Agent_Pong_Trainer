# pong_arena_adapter.py - Pong-Specific Adapter for Agent Byte Integration
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time
from environment_adapter import EnvironmentAdapter, UniversalStateNormalizer, StandardizedStateDimensions


class PongArenaAdapter(EnvironmentAdapter):
    """
    Pong-Specific Adapter for Agent Byte Integration

    This adapter translates between Agent Byte's universal AI system and the
    Arena Pong environment. It handles all transfer learning logic while keeping
    the Pong environment pure and modular.
    """

    def __init__(self, pong_environment):
        """
        Initialize the Pong adapter with a specific Pong environment instance

        Args:
            pong_environment: Instance of ArenaPongEnvironment (pure game logic)
        """
        self.pong_env = pong_environment
        self.environment_id = 'arena_pong'
        self.environment_type = 'real_time_competitive'

        # Initialize universal adapter components
        self.universal_normalizer = UniversalStateNormalizer()
        self.dimensions = StandardizedStateDimensions()

        # Pong-specific metadata for universal adapter
        self.environment_metadata = {
            'environment_id': self.environment_id,
            'environment_type': self.environment_type,
            'entity_positions': [0, 1, 4, 5],  # ball_x, ball_y, player_paddle_y, ai_paddle_y
            'movement_indicators': [2, 3],  # ball_dx, ball_dy
            'timing_features': [12, 13],  # game_active, time_based_features
            'strategic_indicators': [6, 7, 11],  # score_diff, success_streak, ball_direction
            'performance_indicators': [8, 9, 10],  # success_rate, momentum, distance
            'action_mappings': {
                0: 'move_up',
                1: 'stay',
                2: 'move_down'
            },
            'transferable_skills': [
                'trajectory_prediction',
                'timing_optimization',
                'strategic_positioning',
                'pattern_recognition',
                'error_recovery',
                'adaptive_strategy'
            ]
        }

        print(f"🔗 Pong Arena Adapter initialized for {self.environment_id}")

    # ========================================
    # UNIVERSAL ADAPTER INTERFACE IMPLEMENTATION
    # ========================================

    def normalize_state_to_256d(self, raw_state: np.ndarray) -> np.ndarray:
        """
        Convert Pong's 14-dimension state to universal 256-dimension format

        This is the key method that enables Agent Byte to work with Pong while
        maintaining compatibility with all other environments.
        """
        return self.universal_normalizer.normalize_any_state(
            raw_state,
            self.environment_type,
            self.environment_metadata
        )

    def get_transfer_context(self) -> Dict[str, Any]:
        """
        Define what transferable skills Pong teaches

        This is the knowledge that can transfer to Chess, Trading, Robotics, etc.
        """
        return {
            'environment_id': self.environment_id,
            'environment_type': self.environment_type,

            # Universal transferable skills that Pong teaches
            'transferable_skills': [
                {
                    'skill_id': 'trajectory_prediction',
                    'skill_name': 'Dynamic Object Trajectory Prediction',
                    'description': 'Ability to predict the future position of moving objects',
                    'pong_implementation': 'Predict ball path for optimal paddle positioning',
                    'universal_applications': [
                        'Chess: Predict opponent piece movement patterns',
                        'Trading: Predict price movement trends and trajectories',
                        'Robotics: Predict object motion for grasping and avoidance'
                    ],
                    'confidence_level': 0.9,
                    'measurement_metric': 'ball_prediction_accuracy'
                },
                {
                    'skill_id': 'timing_optimization',
                    'skill_name': 'Optimal Timing Execution',
                    'description': 'Ability to execute actions at the optimal moment',
                    'pong_implementation': 'Time paddle movements to intercept ball perfectly',
                    'universal_applications': [
                        'Chess: Optimal move timing under time pressure',
                        'Trading: Optimal entry and exit timing in markets',
                        'Robotics: Optimal action execution timing for efficiency'
                    ],
                    'confidence_level': 0.85,
                    'measurement_metric': 'hit_timing_precision'
                },
                {
                    'skill_id': 'strategic_positioning',
                    'skill_name': 'Strategic Positioning Optimization',
                    'description': 'Ability to position optimally for maximum advantage',
                    'pong_implementation': 'Position paddle optimally for defense and offense',
                    'universal_applications': [
                        'Chess: Optimal piece positioning for board control',
                        'Trading: Optimal portfolio positioning for risk/reward',
                        'Robotics: Optimal robot positioning for task execution'
                    ],
                    'confidence_level': 0.8,
                    'measurement_metric': 'positioning_effectiveness'
                },
                {
                    'skill_id': 'pattern_recognition',
                    'skill_name': 'Pattern Recognition and Adaptation',
                    'description': 'Ability to recognize patterns and adapt behavior accordingly',
                    'pong_implementation': 'Recognize opponent patterns and ball behavior',
                    'universal_applications': [
                        'Chess: Recognize opening, middle, and endgame patterns',
                        'Trading: Recognize market cycles and trading patterns',
                        'Robotics: Recognize environmental patterns and obstacles'
                    ],
                    'confidence_level': 0.75,
                    'measurement_metric': 'pattern_recognition_score'
                },
                {
                    'skill_id': 'error_recovery',
                    'skill_name': 'Error Recovery and Continuation',
                    'description': 'Ability to recover from mistakes and continue effectively',
                    'pong_implementation': 'Recover from missed hits and maintain performance',
                    'universal_applications': [
                        'Chess: Recover from bad moves with strategic adaptation',
                        'Trading: Recover from losses with improved risk management',
                        'Robotics: Recover from failed actions and retry with adjustments'
                    ],
                    'confidence_level': 0.7,
                    'measurement_metric': 'recovery_speed_after_miss'
                },
                {
                    'skill_id': 'adaptive_strategy',
                    'skill_name': 'Adaptive Strategy Development',
                    'description': 'Ability to adapt strategy based on changing conditions',
                    'pong_implementation': 'Adapt playing style based on opponent behavior',
                    'universal_applications': [
                        'Chess: Adapt strategy based on opponent playing style',
                        'Trading: Adapt strategy based on market conditions',
                        'Robotics: Adapt behavior based on environmental changes'
                    ],
                    'confidence_level': 0.8,
                    'measurement_metric': 'strategy_adaptation_speed'
                }
            ],

            # Abstract concepts that transfer universally
            'abstract_concepts': [
                {
                    'concept': 'moving_object_interception',
                    'description': 'Intercepting moving targets requires prediction and positioning',
                    'universality': 'Applies to any domain with moving elements'
                },
                {
                    'concept': 'spatial_temporal_coordination',
                    'description': 'Coordinating actions across space and time for optimal outcomes',
                    'universality': 'Fundamental to most decision-making domains'
                },
                {
                    'concept': 'predictive_behavior_modeling',
                    'description': 'Modeling and predicting future states for decision making',
                    'universality': 'Core to strategic thinking across domains'
                },
                {
                    'concept': 'competitive_decision_making',
                    'description': 'Making decisions in competitive/adversarial environments',
                    'universality': 'Applies to markets, games, negotiations, and conflicts'
                }
            ],

            # Success patterns that transfer
            'success_patterns': [
                {
                    'pattern': 'early_positioning_beats_reactive_movement',
                    'description': 'Proactive positioning is more effective than reactive responses',
                    'transfer_domains': [
                        'Chess: Early piece development',
                        'Trading: Market positioning before trends',
                        'Robotics: Anticipatory movement planning'
                    ]
                },
                {
                    'pattern': 'consistent_prediction_over_lucky_guesses',
                    'description': 'Systematic prediction methods outperform random actions',
                    'transfer_domains': [
                        'Chess: Calculated moves vs intuition',
                        'Trading: Technical analysis vs speculation',
                        'Robotics: Planned vs random motion'
                    ]
                },
                {
                    'pattern': 'adaptive_timing_based_on_feedback',
                    'description': 'Timing should adapt based on environmental feedback',
                    'transfer_domains': [
                        'Chess: Tempo adjustment based on position',
                        'Trading: Market timing based on volatility',
                        'Robotics: Dynamic timing based on environment'
                    ]
                }
            ],

            # Learning objectives that develop transferable skills
            'learning_objectives': [
                {
                    'objective': 'Develop precise trajectory prediction',
                    'measurable_outcome': 'Hit rate improvement and prediction accuracy',
                    'transferable_benefit': 'Enhanced forecasting ability across domains'
                },
                {
                    'objective': 'Master optimal positioning strategies',
                    'measurable_outcome': 'Strategic positioning effectiveness score',
                    'transferable_benefit': 'Improved strategic positioning in any environment'
                },
                {
                    'objective': 'Build robust error recovery mechanisms',
                    'measurable_outcome': 'Recovery speed after mistakes',
                    'transferable_benefit': 'Resilience and adaptability across challenges'
                }
            ]
        }

    def map_abstract_action(self, strategy: str, state: np.ndarray) -> int:
        """
        Convert abstract strategy to Pong-specific action

        This enables skills learned in other environments to be applied in Pong,
        and allows Pong strategies to be expressed in abstract terms.
        """
        # Extract relevant state information for decision making
        ball_x_norm = state[0] if len(state) > 0 else 0
        ball_y_norm = state[1] if len(state) > 1 else 0
        ball_dx_norm = state[2] if len(state) > 2 else 0
        ball_dy_norm = state[3] if len(state) > 3 else 0
        ai_paddle_y_norm = state[5] if len(state) > 5 else 0

        # Map abstract strategies to Pong actions
        if strategy in ['move_toward_target', 'trajectory_prediction', 'optimal_positioning']:
            # Predict where ball will be and move toward it
            if ball_dx_norm > 0:  # Ball moving toward AI
                future_ball_y = ball_y_norm + ball_dy_norm * 0.3  # Predict future position
                paddle_ball_diff = future_ball_y - ai_paddle_y_norm

                if paddle_ball_diff > 0.1:
                    return 2  # Move down
                elif paddle_ball_diff < -0.1:
                    return 0  # Move up
                else:
                    return 1  # Stay
            else:
                return 1  # Stay if ball moving away

        elif strategy in ['defensive_positioning', 'error_recovery']:
            # Take conservative defensive position (center)
            if ai_paddle_y_norm > 0.1:
                return 0  # Move up toward center
            elif ai_paddle_y_norm < -0.1:
                return 2  # Move down toward center
            else:
                return 1  # Stay at center

        elif strategy in ['aggressive_advancement', 'pattern_based_prediction']:
            # Aggressive positioning based on ball pattern
            if ball_dy_norm > 0:  # Ball moving down
                return 2  # Move down to intercept
            elif ball_dy_norm < 0:  # Ball moving up
                return 0  # Move up to intercept
            else:
                return 1  # Stay if no clear pattern

        elif strategy in ['adaptive_strategy', 'timing_optimization']:
            # Adaptive timing-based action
            ball_paddle_distance = abs(ball_y_norm - ai_paddle_y_norm)

            if ball_paddle_distance > 0.2:  # Far from ball
                # Move toward ball
                if ball_y_norm > ai_paddle_y_norm:
                    return 2  # Move down
                else:
                    return 0  # Move up
            else:
                # Close to ball - fine adjustments
                if ball_dy_norm > 0.1:
                    return 2  # Follow ball down
                elif ball_dy_norm < -0.1:
                    return 0  # Follow ball up
                else:
                    return 1  # Stay

        else:
            # Default fallback - basic positioning
            paddle_ball_diff = ball_y_norm - ai_paddle_y_norm
            if paddle_ball_diff > 0.1:
                return 2  # Move down
            elif paddle_ball_diff < -0.1:
                return 0  # Move up
            else:
                return 1  # Stay

    def get_environment_metadata(self) -> Dict[str, Any]:
        """
        Get comprehensive metadata about Pong for transfer learning

        This provides Agent Byte with all the information needed to understand
        how to work with Pong and extract transferable knowledge.
        """
        # Get current performance data from the Pong environment
        if hasattr(self.pong_env, 'get_pong_stats'):
            current_stats = self.pong_env.get_pong_stats()
        else:
            current_stats = {}

        if hasattr(self.pong_env, 'get_transfer_learning_metrics'):
            transfer_metrics = self.pong_env.get_transfer_learning_metrics()
        else:
            transfer_metrics = {}

        return {
            **self.environment_metadata,
            'current_session': {
                'match_id': getattr(self.pong_env, 'match_id', 'unknown'),
                'running': getattr(self.pong_env, 'running', False),
                'duration': getattr(self.pong_env, 'match_duration', 0.0),
                'is_arena_match': getattr(self.pong_env, 'is_arena_match', False)
            },
            'performance_data': current_stats,
            'transfer_readiness': transfer_metrics,
            'state_dimensions': {
                'raw_size': 14,  # Pong's native state size
                'standardized_size': 256,  # Universal standard
                'normalization_mapping': self.environment_metadata
            },
            'skill_development_indicators': self._calculate_skill_development()
        }

    def _calculate_skill_development(self) -> Dict[str, float]:
        """Calculate how well each transferable skill is being developed"""
        skill_scores = {}

        # Get performance metrics from Pong environment
        if hasattr(self.pong_env, 'performance_metrics'):
            metrics = self.pong_env.performance_metrics

            # Trajectory prediction skill
            if metrics.get('prediction_accuracy'):
                skill_scores['trajectory_prediction'] = np.mean(metrics['prediction_accuracy'])
            else:
                skill_scores['trajectory_prediction'] = 0.0

            # Positioning skill
            best_streak = getattr(self.pong_env, 'best_success_streak', 0)
            skill_scores['strategic_positioning'] = min(1.0, best_streak / 20.0)

            # Timing skill
            hit_rate = 0.0
            if hasattr(self.pong_env, 'task_successes') and hasattr(self.pong_env, 'task_failures'):
                total_attempts = self.pong_env.task_successes + self.pong_env.task_failures
                if total_attempts > 0:
                    hit_rate = self.pong_env.task_successes / total_attempts
            skill_scores['timing_optimization'] = hit_rate

            # Pattern recognition skill
            if hasattr(self.pong_env, 'action_history'):
                action_diversity = len(set(
                    a.get('action', 0) for a in self.pong_env.action_history[-20:]
                )) / 3.0 if self.pong_env.action_history else 0
                skill_scores['pattern_recognition'] = action_diversity
            else:
                skill_scores['pattern_recognition'] = 0.0

            # Error recovery skill (based on streak recovery)
            skill_scores['error_recovery'] = min(1.0, best_streak / 10.0)

            # Adaptive strategy skill (based on overall performance improvement)
            skill_scores['adaptive_strategy'] = skill_scores.get('strategic_positioning', 0.0)

        else:
            # Default values if no metrics available
            for skill in self.environment_metadata['transferable_skills']:
                skill_scores[skill] = 0.0

        return skill_scores

    # ========================================
    # PONG-SPECIFIC ADAPTER METHODS
    # ========================================

    def create_pong_state(self) -> np.ndarray:
        """
        Get current Pong state in raw format

        This delegates to the pure Pong environment to get the game state,
        maintaining separation between adapter and environment.
        """
        if hasattr(self.pong_env, 'create_state'):
            return self.pong_env.create_state()
        else:
            # Fallback empty state
            return np.zeros(14, dtype=np.float32)

    def execute_pong_action(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute action in Pong environment

        This delegates to the pure Pong environment while providing a consistent
        interface for Agent Byte.
        """
        if hasattr(self.pong_env, 'step'):
            return self.pong_env.step(action)
        else:
            # Fallback
            return self.create_pong_state(), 0.0, False

    def reset_pong_environment(self):
        """Reset the Pong environment through the adapter"""
        if hasattr(self.pong_env, 'reset_game'):
            self.pong_env.reset_game()

    def get_pong_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary with transfer learning insights

        This combines Pong-specific performance data with transfer learning
        readiness indicators.
        """
        # Get basic Pong stats
        if hasattr(self.pong_env, 'get_pong_stats'):
            pong_stats = self.pong_env.get_pong_stats()
        else:
            pong_stats = {}

        # Calculate transfer learning insights
        skill_development = self._calculate_skill_development()
        transfer_context = self.get_transfer_context()

        # Calculate overall transfer readiness
        skill_scores = list(skill_development.values())
        transfer_readiness = np.mean(skill_scores) if skill_scores else 0.0

        return {
            'pong_performance': pong_stats,
            'skill_development': skill_development,
            'transfer_readiness': transfer_readiness,
            'transferable_skills_count': len(transfer_context['transferable_skills']),
            'most_developed_skills': [
                                         skill for skill, score in sorted(skill_development.items(),
                                                                          key=lambda x: x[1], reverse=True)
                                     ][:3],
            'recommended_transfer_environments': self._get_transfer_recommendations(),
            'adapter_metadata': {
                'environment_id': self.environment_id,
                'adapter_version': '1.0.0',
                'skills_mapped': len(self.environment_metadata['transferable_skills'])
            }
        }

    def _get_transfer_recommendations(self) -> List[str]:
        """Get recommended environments for transfer based on Pong skills developed"""
        skill_development = self._calculate_skill_development()
        recommendations = []

        # Recommend based on strongest developed skills
        if skill_development.get('trajectory_prediction', 0) > 0.7:
            recommendations.extend(['chess', 'trading', 'robotics'])

        if skill_development.get('timing_optimization', 0) > 0.7:
            recommendations.extend(['rhythm_games', 'trading', 'real_time_strategy'])

        if skill_development.get('strategic_positioning', 0) > 0.7:
            recommendations.extend(['chess', 'go', 'military_strategy'])

        if skill_development.get('pattern_recognition', 0) > 0.7:
            recommendations.extend(['pattern_matching', 'anomaly_detection', 'forecasting'])

        # Remove duplicates and return top recommendations
        return list(dict.fromkeys(recommendations))[:5]


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Pong Arena Adapter...")


    # Mock Pong environment for testing
    class MockPongEnvironment:
        def __init__(self):
            self.match_id = "test_match"
            self.running = True
            self.match_duration = 120.0
            self.is_arena_match = True
            self.task_successes = 15
            self.task_failures = 5
            self.best_success_streak = 8
            self.performance_metrics = {
                'prediction_accuracy': [0.8, 0.75, 0.9, 0.85],
                'reaction_times': [0.1, 0.12, 0.09]
            }
            self.action_history = [
                {'action': 0}, {'action': 1}, {'action': 2}, {'action': 1}
            ]

        def create_state(self):
            return np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4])

        def step(self, action):
            return self.create_state(), 1.0, False

        def get_pong_stats(self):
            return {'hit_rate': 75.0, 'total_hits': 15, 'total_misses': 5}

        def get_transfer_learning_metrics(self):
            return {'transfer_readiness': 0.8}


    # Test the adapter
    mock_env = MockPongEnvironment()
    adapter = PongArenaAdapter(mock_env)

    # Test universal adapter methods
    print(f"\n🔄 Testing Universal Adapter Interface:")

    # Test state normalization
    raw_state = adapter.create_pong_state()
    normalized_state = adapter.normalize_state_to_256d(raw_state)
    print(f"   State normalization: {len(raw_state)} → {len(normalized_state)} dimensions")

    # Test transfer context
    transfer_context = adapter.get_transfer_context()
    print(f"   Transfer skills: {len(transfer_context['transferable_skills'])}")

    # Test abstract action mapping
    test_strategies = ['trajectory_prediction', 'defensive_positioning', 'adaptive_strategy']
    print(f"\n🎯 Testing Abstract Action Mapping:")
    for strategy in test_strategies:
        action = adapter.map_abstract_action(strategy, raw_state)
        action_name = ['move_up', 'stay', 'move_down'][action]
        print(f"   {strategy} → {action_name}")

    # Test environment metadata
    metadata = adapter.get_environment_metadata()
    print(f"\n📊 Environment Metadata:")
    print(f"   Environment ID: {metadata['environment_id']}")
    print(f"   Transferable skills: {len(metadata['transferable_skills'])}")
    print(f"   Current session: {metadata['current_session']['match_id']}")

    # Test performance summary
    performance = adapter.get_pong_performance_summary()
    print(f"\n📈 Performance Summary:")
    print(f"   Transfer readiness: {performance['transfer_readiness']:.2f}")
    print(f"   Most developed skills: {performance['most_developed_skills']}")
    print(f"   Recommended environments: {performance['recommended_transfer_environments']}")

    print(f"\n✅ Pong Arena Adapter test complete!")
    print(f"🔗 Adapter successfully bridges Agent Byte ↔ Pong Environment")
    print(f"🌟 Transfer learning ready for Chess, Trading, Robotics, and more!")