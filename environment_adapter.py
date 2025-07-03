# environment_adapter.py - Universal Environment Abstraction Layer for Transfer Learning
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import json
import time


class EnvironmentAdapter(ABC):
    """
    Universal Environment Adapter Interface

    This is the contract that ALL environments must implement to work with
    Agent Byte's transferable AI system. Think of it as USB for AI environments.
    """

    @abstractmethod
    def normalize_state_to_256d(self, raw_state: np.ndarray) -> np.ndarray:
        """
        Convert ANY environment state to standard 256-dimension input

        This is the KEY method that enables transfer learning across environments.
        All environments must convert their unique state to this standard format.
        """
        pass

    @abstractmethod
    def get_transfer_context(self) -> Dict[str, Any]:
        """
        Define what transferable skills this environment teaches

        Returns context about what skills learned here can transfer to other domains.
        """
        pass

    @abstractmethod
    def map_abstract_action(self, strategy: str, state: np.ndarray) -> int:
        """
        Convert abstract strategy to environment-specific action

        Args:
            strategy: Abstract strategy like "move_toward_target" or "defensive_positioning"
            state: Current environment state for context

        Returns:
            Environment-specific action index
        """
        pass

    @abstractmethod
    def get_environment_metadata(self) -> Dict[str, Any]:
        """Get metadata about this environment for transfer learning"""
        pass


class StandardizedStateDimensions:
    """
    Standardized 256-dimension layout for cross-environment compatibility

    This ensures that similar concepts map to similar positions across ALL environments,
    enabling effective transfer learning.
    """

    # Core positions (0-31): Object/Entity positions and orientations
    ENTITY_POSITIONS_START = 0
    ENTITY_POSITIONS_END = 31

    # Movement vectors (32-63): Velocities, directions, momentum
    MOVEMENT_VECTORS_START = 32
    MOVEMENT_VECTORS_END = 63

    # Timing features (64-95): Temporal patterns, rhythms, sequences
    TIMING_FEATURES_START = 64
    TIMING_FEATURES_END = 95

    # Strategic context (96-127): Goals, objectives, competitive state
    STRATEGIC_CONTEXT_START = 96
    STRATEGIC_CONTEXT_END = 127

    # Performance metrics (128-159): Success rates, efficiency measures
    PERFORMANCE_METRICS_START = 128
    PERFORMANCE_METRICS_END = 159

    # Pattern recognition (160-191): Historical patterns, trend analysis
    PATTERN_RECOGNITION_START = 160
    PATTERN_RECOGNITION_END = 191

    # Meta-learning indicators (192-223): Learning progress, adaptation signals
    META_LEARNING_START = 192
    META_LEARNING_END = 223

    # Environment-specific (224-255): Domain-specific features
    ENVIRONMENT_SPECIFIC_START = 224
    ENVIRONMENT_SPECIFIC_END = 255


class TransferableSkillMapper:
    """
    Maps abstract skills to environment-specific implementations

    This is the "universal translator" that allows skills learned in one environment
    to be applied in completely different domains.
    """

    def __init__(self):
        # Universal skill mappings across environments
        self.skill_mappings = {
            # Moving Object Prediction (transfers universally)
            "trajectory_prediction": {
                'pong': "predict ball path for paddle positioning",
                'chess': "predict opponent piece movement patterns",
                'trading': "predict price movement trends",
                'flappy_bird': "predict pipe gap timing",
                'robotics': "predict object motion for grasping"
            },

            # Optimal Timing (universal skill)
            "timing_optimization": {
                'pong': "optimal paddle movement timing",
                'chess': "optimal move timing in time pressure",
                'trading': "optimal entry/exit timing",
                'flappy_bird': "optimal jump timing",
                'robotics': "optimal action execution timing"
            },

            # Strategic Positioning (spatial intelligence)
            "strategic_positioning": {
                'pong': "optimal paddle positioning for defense/offense",
                'chess': "optimal piece positioning for board control",
                'trading': "optimal portfolio positioning",
                'flappy_bird': "optimal bird positioning in flight path",
                'robotics': "optimal robot positioning for task execution"
            },

            # Pattern Recognition (meta-cognitive skill)
            "pattern_recognition": {
                'pong': "recognize opponent hitting patterns",
                'chess': "recognize opening/endgame patterns",
                'trading': "recognize market cycle patterns",
                'flappy_bird': "recognize pipe spacing patterns",
                'robotics': "recognize environmental patterns"
            },

            # Error Recovery (resilience skill)
            "error_recovery": {
                'pong': "recover from missed hits and continue play",
                'chess': "recover from bad moves with strategic adaptation",
                'trading': "recover from losses with risk management",
                'flappy_bird': "recover from near-collisions with quick adjustments",
                'robotics': "recover from failed actions and retry"
            },

            # Adaptive Strategy (high-level intelligence)
            "adaptive_strategy": {
                'pong': "adapt playing style based on opponent behavior",
                'chess': "adapt strategy based on opponent's playing style",
                'trading': "adapt strategy based on market conditions",
                'flappy_bird': "adapt flight pattern based on pipe configurations",
                'robotics': "adapt behavior based on environmental changes"
            }
        }

        # Action mapping templates
        self.action_templates = {
            "move_toward_target": {
                'description': "Move toward primary objective/target",
                'requires_state': ['target_position', 'current_position']
            },
            "defensive_positioning": {
                'description': "Take defensive/conservative position",
                'requires_state': ['threat_assessment', 'safe_zones']
            },
            "aggressive_advancement": {
                'description': "Take offensive/aggressive action",
                'requires_state': ['opportunity_assessment', 'risk_tolerance']
            },
            "pattern_based_prediction": {
                'description': "Act based on recognized patterns",
                'requires_state': ['historical_patterns', 'current_context']
            },
            "error_correction": {
                'description': "Correct recent mistakes or adapt strategy",
                'requires_state': ['recent_performance', 'error_indicators']
            }
        }

    def get_transferable_skills_for_environment(self, environment_id: str) -> List[str]:
        """Get all transferable skills that apply to this environment"""
        applicable_skills = []
        for skill, env_mappings in self.skill_mappings.items():
            if environment_id in env_mappings:
                applicable_skills.append(skill)
        return applicable_skills

    def translate_skill_to_environment(self, skill: str, from_env: str, to_env: str) -> Optional[str]:
        """Translate a skill from one environment to another"""
        if skill in self.skill_mappings:
            env_mappings = self.skill_mappings[skill]
            if from_env in env_mappings and to_env in env_mappings:
                return {
                    'source_implementation': env_mappings[from_env],
                    'target_implementation': env_mappings[to_env],
                    'transfer_confidence': 0.8,  # Could be learned over time
                    'adaptation_required': from_env != to_env
                }
        return None

    def get_cross_environment_correlation(self, env1: str, env2: str) -> float:
        """Calculate how much skill transfer potential exists between two environments"""
        shared_skills = 0
        total_skills = 0

        for skill, env_mappings in self.skill_mappings.items():
            total_skills += 1
            if env1 in env_mappings and env2 in env_mappings:
                shared_skills += 1

        return shared_skills / max(1, total_skills)


class UniversalStateNormalizer:
    """
    Handles conversion of any environment state to the standardized 256-dimension format

    This is the core system that enables transfer learning by ensuring all environments
    speak the same "language" to the neural network.
    """

    def __init__(self):
        self.dimensions = StandardizedStateDimensions()

    def normalize_any_state(self, raw_state: np.ndarray, environment_type: str,
                            environment_metadata: Dict[str, Any]) -> np.ndarray:
        """
        Universal state normalization that works for ANY environment

        Args:
            raw_state: Environment's native state representation
            environment_type: Type of environment (pong, chess, trading, etc.)
            environment_metadata: Metadata about the environment structure

        Returns:
            Standardized 256-dimension state vector
        """
        normalized_state = np.zeros(256, dtype=np.float32)

        # Step 1: Extract universal concepts from raw state
        universal_features = self._extract_universal_features(raw_state, environment_type, environment_metadata)

        # Step 2: Map to standardized dimensions
        self._map_to_standard_dimensions(universal_features, normalized_state)

        # Step 3: Add positional encoding for unused dimensions
        self._add_positional_encoding(normalized_state, len(raw_state))

        # Step 4: Ensure all values are properly bounded
        normalized_state = np.clip(normalized_state, -2.0, 2.0)

        return normalized_state

    def _extract_universal_features(self, raw_state: np.ndarray, environment_type: str,
                                    metadata: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract universal concepts from environment-specific state"""
        features = {}

        # Entity positions (objects, players, pieces, etc.)
        if 'entity_positions' in metadata:
            positions = self._extract_positions(raw_state, metadata['entity_positions'])
            features['entity_positions'] = positions

        # Movement vectors (velocities, directions, trends)
        if 'movement_indicators' in metadata:
            movements = self._extract_movements(raw_state, metadata['movement_indicators'])
            features['movement_vectors'] = movements

        # Timing information (sequences, rhythms, temporal patterns)
        if 'timing_features' in metadata:
            timing = self._extract_timing(raw_state, metadata['timing_features'])
            features['timing_features'] = timing

        # Strategic context (goals, objectives, competitive state)
        if 'strategic_indicators' in metadata:
            strategy = self._extract_strategy(raw_state, metadata['strategic_indicators'])
            features['strategic_context'] = strategy

        # Performance metrics (success rates, scores, efficiency)
        if 'performance_indicators' in metadata:
            performance = self._extract_performance(raw_state, metadata['performance_indicators'])
            features['performance_metrics'] = performance

        return features

    def _extract_positions(self, raw_state: np.ndarray, position_indices: List[int]) -> np.ndarray:
        """Extract position information from raw state"""
        positions = np.zeros(32)  # Max 32 position features
        for i, idx in enumerate(position_indices[:16]):  # Max 16 position pairs (x,y)
            if idx < len(raw_state):
                positions[i * 2] = raw_state[idx]  # X coordinate
                if idx + 1 < len(raw_state):
                    positions[i * 2 + 1] = raw_state[idx + 1]  # Y coordinate
        return positions

    def _extract_movements(self, raw_state: np.ndarray, movement_indices: List[int]) -> np.ndarray:
        """Extract movement/velocity information from raw state"""
        movements = np.zeros(32)  # Max 32 movement features
        for i, idx in enumerate(movement_indices[:16]):  # Max 16 movement pairs
            if idx < len(raw_state):
                movements[i * 2] = raw_state[idx]  # X velocity
                if idx + 1 < len(raw_state):
                    movements[i * 2 + 1] = raw_state[idx + 1]  # Y velocity
        return movements

    def _extract_timing(self, raw_state: np.ndarray, timing_indices: List[int]) -> np.ndarray:
        """Extract timing and temporal pattern information"""
        timing = np.zeros(32)  # Max 32 timing features
        for i, idx in enumerate(timing_indices[:32]):
            if idx < len(raw_state):
                timing[i] = raw_state[idx]
        return timing

    def _extract_strategy(self, raw_state: np.ndarray, strategy_indices: List[int]) -> np.ndarray:
        """Extract strategic context information"""
        strategy = np.zeros(32)  # Max 32 strategic features
        for i, idx in enumerate(strategy_indices[:32]):
            if idx < len(raw_state):
                strategy[i] = raw_state[idx]
        return strategy

    def _extract_performance(self, raw_state: np.ndarray, performance_indices: List[int]) -> np.ndarray:
        """Extract performance and success metrics"""
        performance = np.zeros(32)  # Max 32 performance features
        for i, idx in enumerate(performance_indices[:32]):
            if idx < len(raw_state):
                performance[i] = raw_state[idx]
        return performance

    def _map_to_standard_dimensions(self, features: Dict[str, np.ndarray], normalized_state: np.ndarray):
        """Map extracted features to standardized dimension ranges"""
        dims = self.dimensions

        # Map entity positions (0-31)
        if 'entity_positions' in features:
            end_idx = min(dims.ENTITY_POSITIONS_END + 1,
                          dims.ENTITY_POSITIONS_START + len(features['entity_positions']))
            normalized_state[dims.ENTITY_POSITIONS_START:end_idx] = features['entity_positions'][
                                                                    :end_idx - dims.ENTITY_POSITIONS_START]

        # Map movement vectors (32-63)
        if 'movement_vectors' in features:
            start = dims.MOVEMENT_VECTORS_START
            end_idx = min(dims.MOVEMENT_VECTORS_END + 1, start + len(features['movement_vectors']))
            normalized_state[start:end_idx] = features['movement_vectors'][:end_idx - start]

        # Map timing features (64-95)
        if 'timing_features' in features:
            start = dims.TIMING_FEATURES_START
            end_idx = min(dims.TIMING_FEATURES_END + 1, start + len(features['timing_features']))
            normalized_state[start:end_idx] = features['timing_features'][:end_idx - start]

        # Map strategic context (96-127)
        if 'strategic_context' in features:
            start = dims.STRATEGIC_CONTEXT_START
            end_idx = min(dims.STRATEGIC_CONTEXT_END + 1, start + len(features['strategic_context']))
            normalized_state[start:end_idx] = features['strategic_context'][:end_idx - start]

        # Map performance metrics (128-159)
        if 'performance_metrics' in features:
            start = dims.PERFORMANCE_METRICS_START
            end_idx = min(dims.PERFORMANCE_METRICS_END + 1, start + len(features['performance_metrics']))
            normalized_state[start:end_idx] = features['performance_metrics'][:end_idx - start]

    def _add_positional_encoding(self, normalized_state: np.ndarray, original_state_size: int):
        """Add positional encoding to unused dimensions"""
        for i in range(256):
            if normalized_state[i] == 0.0:  # Unused dimension
                # Add small positional encoding to provide information about state structure
                normalized_state[i] = np.sin(i * 0.01 + original_state_size * 0.001) * 0.1


class CrossEnvironmentKnowledgeTransfer:
    """
    Handles knowledge transfer between different environments

    This system tracks what skills work where and facilitates the transfer
    of successful strategies across domains.
    """

    def __init__(self):
        self.skill_mapper = TransferableSkillMapper()
        self.transfer_history = []
        self.environment_correlations = {}
        self.skill_effectiveness_matrix = {}

    def initiate_transfer(self, agent_id: str, from_environment: str, to_environment: str,
                          source_skills: List[str]) -> Dict[str, Any]:
        """
        Initiate knowledge transfer from one environment to another

        Args:
            agent_id: ID of the agent performing transfer
            from_environment: Source environment ID
            to_environment: Target environment ID
            source_skills: List of skills to transfer

        Returns:
            Transfer plan with mappings and confidence scores
        """
        transfer_plan = {
            'transfer_id': f"{agent_id}_{from_environment}_to_{to_environment}_{int(time.time())}",
            'agent_id': agent_id,
            'from_environment': from_environment,
            'to_environment': to_environment,
            'initiated_at': time.time(),
            'skill_transfers': [],
            'overall_confidence': 0.0,
            'adaptation_requirements': []
        }

        successful_transfers = 0

        for skill in source_skills:
            skill_transfer = self.skill_mapper.translate_skill_to_environment(
                skill, from_environment, to_environment
            )

            if skill_transfer:
                transfer_plan['skill_transfers'].append({
                    'skill': skill,
                    'source_implementation': skill_transfer['source_implementation'],
                    'target_implementation': skill_transfer['target_implementation'],
                    'confidence': skill_transfer['transfer_confidence'],
                    'adaptation_required': skill_transfer['adaptation_required']
                })
                successful_transfers += 1

                if skill_transfer['adaptation_required']:
                    transfer_plan['adaptation_requirements'].append({
                        'skill': skill,
                        'adaptation_type': 'implementation_mapping',
                        'description': f"Adapt {skill} from {from_environment} context to {to_environment} context"
                    })

        # Calculate overall confidence
        if source_skills:
            transfer_plan['overall_confidence'] = successful_transfers / len(source_skills)

        # Store transfer attempt
        self.transfer_history.append(transfer_plan)

        return transfer_plan

    def record_transfer_outcome(self, transfer_id: str, success_rate: float,
                                performance_improvement: float):
        """Record the outcome of a transfer learning attempt"""
        for transfer in self.transfer_history:
            if transfer['transfer_id'] == transfer_id:
                transfer['completed_at'] = time.time()
                transfer['success_rate'] = success_rate
                transfer['performance_improvement'] = performance_improvement

                # Update skill effectiveness matrix
                for skill_transfer in transfer['skill_transfers']:
                    skill = skill_transfer['skill']
                    env_pair = f"{transfer['from_environment']}_to_{transfer['to_environment']}"

                    if skill not in self.skill_effectiveness_matrix:
                        self.skill_effectiveness_matrix[skill] = {}

                    self.skill_effectiveness_matrix[skill][env_pair] = {
                        'success_rate': success_rate,
                        'performance_improvement': performance_improvement,
                        'transfer_confidence': skill_transfer['confidence']
                    }
                break

    def get_transfer_recommendations(self, agent_id: str, current_environment: str,
                                     available_environments: List[str]) -> List[Dict[str, Any]]:
        """Get recommendations for which environments would benefit from transfer learning"""
        recommendations = []

        for target_env in available_environments:
            if target_env != current_environment:
                correlation = self.skill_mapper.get_cross_environment_correlation(
                    current_environment, target_env
                )

                applicable_skills = self.skill_mapper.get_transferable_skills_for_environment(target_env)

                recommendation = {
                    'target_environment': target_env,
                    'correlation_score': correlation,
                    'transferable_skills_count': len(applicable_skills),
                    'transferable_skills': applicable_skills,
                    'transfer_confidence': correlation * 0.8,  # Conservative estimate
                    'recommended': correlation > 0.5
                }

                recommendations.append(recommendation)

        # Sort by correlation score
        recommendations.sort(key=lambda x: x['correlation_score'], reverse=True)

        return recommendations

    def get_transfer_history_summary(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of transfer learning history"""
        relevant_transfers = [
            t for t in self.transfer_history
            if agent_id is None or t['agent_id'] == agent_id
        ]

        if not relevant_transfers:
            return {'total_transfers': 0, 'average_success_rate': 0.0}

        total_transfers = len(relevant_transfers)
        completed_transfers = [t for t in relevant_transfers if 'success_rate' in t]

        if completed_transfers:
            avg_success_rate = np.mean([t['success_rate'] for t in completed_transfers])
            avg_improvement = np.mean([t['performance_improvement'] for t in completed_transfers])
        else:
            avg_success_rate = 0.0
            avg_improvement = 0.0

        return {
            'total_transfers': total_transfers,
            'completed_transfers': len(completed_transfers),
            'average_success_rate': avg_success_rate,
            'average_performance_improvement': avg_improvement,
            'most_transferred_skills': self._get_most_transferred_skills(relevant_transfers),
            'environment_pairs': self._get_environment_pairs(relevant_transfers)
        }

    def _get_most_transferred_skills(self, transfers: List[Dict]) -> List[Dict[str, Any]]:
        """Get most frequently transferred skills"""
        skill_counts = {}

        for transfer in transfers:
            for skill_transfer in transfer.get('skill_transfers', []):
                skill = skill_transfer['skill']
                skill_counts[skill] = skill_counts.get(skill, 0) + 1

        return [
            {'skill': skill, 'transfer_count': count}
            for skill, count in sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
        ]

    def _get_environment_pairs(self, transfers: List[Dict]) -> List[Dict[str, Any]]:
        """Get most common environment transfer pairs"""
        pair_counts = {}

        for transfer in transfers:
            pair = f"{transfer['from_environment']} → {transfer['to_environment']}"
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        return [
            {'environment_pair': pair, 'transfer_count': count}
            for pair, count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
        ]


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Universal Environment Abstraction Layer...")

    # Test skill mapping
    mapper = TransferableSkillMapper()

    # Test cross-environment correlation
    pong_chess_correlation = mapper.get_cross_environment_correlation('pong', 'chess')
    print(f"\n🔄 Pong ↔ Chess Correlation: {pong_chess_correlation:.2f}")

    # Test skill translation
    trajectory_transfer = mapper.translate_skill_to_environment('trajectory_prediction', 'pong', 'chess')
    print(f"\n🎯 Trajectory Prediction Transfer:")
    print(f"   Pong: {trajectory_transfer['source_implementation']}")
    print(f"   Chess: {trajectory_transfer['target_implementation']}")
    print(f"   Confidence: {trajectory_transfer['transfer_confidence']}")

    # Test universal state normalization
    normalizer = UniversalStateNormalizer()

    # Simulate Pong state
    pong_metadata = {
        'entity_positions': [0, 1, 4, 5],  # ball_x, ball_y, ai_paddle_y, player_paddle_y
        'movement_indicators': [2, 3],  # ball_dx, ball_dy
        'strategic_indicators': [6, 7],  # score_diff, success_streak
        'performance_indicators': [8, 9]  # success_rate, streak_bonus
    }

    test_pong_state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    normalized_pong = normalizer.normalize_any_state(test_pong_state, 'pong', pong_metadata)

    print(f"\n📊 State Normalization Test:")
    print(f"   Original: {len(test_pong_state)} dimensions")
    print(f"   Normalized: {len(normalized_pong)} dimensions")
    print(f"   Entity positions: {normalized_pong[0:4]}")
    print(f"   Movement vectors: {normalized_pong[32:36]}")

    # Test knowledge transfer system
    transfer_system = CrossEnvironmentKnowledgeTransfer()

    # Simulate transfer from Pong to Chess
    pong_skills = ['trajectory_prediction', 'timing_optimization', 'pattern_recognition']
    transfer_plan = transfer_system.initiate_transfer('agent_001', 'pong', 'chess', pong_skills)

    print(f"\n🔄 Transfer Plan: Pong → Chess")
    print(f"   Transfer ID: {transfer_plan['transfer_id']}")
    print(f"   Overall Confidence: {transfer_plan['overall_confidence']:.2f}")
    print(f"   Skills Transferred: {len(transfer_plan['skill_transfers'])}")

    for skill_transfer in transfer_plan['skill_transfers']:
        print(f"   • {skill_transfer['skill']}: {skill_transfer['confidence']:.2f} confidence")

    # Test transfer recommendations
    recommendations = transfer_system.get_transfer_recommendations(
        'agent_001', 'pong', ['chess', 'trading', 'flappy_bird']
    )

    print(f"\n🎯 Transfer Recommendations from Pong:")
    for rec in recommendations:
        print(
            f"   {rec['target_environment']}: {rec['correlation_score']:.2f} correlation, {rec['transferable_skills_count']} skills")

    print(f"\n✅ Universal Environment Abstraction Layer test complete!")
    print(f"🌟 Ready for cross-environment transfer learning!")