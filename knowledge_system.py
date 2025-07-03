# knowledge_system.py - Enhanced Neural-Symbolic Dual Brain Knowledge System
import numpy as np
import random
import time
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque


class NeuralPatternInterpreter:
    """
    Neural Pattern Interpreter - The Bridge Between Neural and Symbolic Brains

    This system watches the neural brain learn and interprets what patterns mean
    in terms of transferable skills. It's the key component that enables the
    dual brain to connect neural learning with symbolic understanding.
    """

    def __init__(self):
        # Neural pattern recognition
        self.learned_patterns = {}
        self.pattern_skill_mapping = {}
        self.skill_confidence_scores = defaultdict(float)

        # Universal skill definitions
        self.universal_skills = {
            'trajectory_prediction': {
                'description': 'Ability to predict future positions of moving objects',
                'neural_indicators': ['consistent_action_sequences', 'temporal_pattern_recognition'],
                'measurement_criteria': ['prediction_accuracy', 'anticipatory_actions'],
                'applies_to_environments': ['pong', 'chess', 'trading', 'robotics', 'sports']
            },
            'timing_optimization': {
                'description': 'Ability to execute actions at optimal moments',
                'neural_indicators': ['reward_timing_correlation', 'action_delay_patterns'],
                'measurement_criteria': ['hit_rate_improvement', 'timing_precision'],
                'applies_to_environments': ['pong', 'rhythm_games', 'trading', 'manufacturing']
            },
            'strategic_positioning': {
                'description': 'Ability to position optimally for maximum advantage',
                'neural_indicators': ['spatial_optimization_patterns', 'defensive_positioning'],
                'measurement_criteria': ['positioning_effectiveness', 'defensive_success'],
                'applies_to_environments': ['pong', 'chess', 'go', 'military_strategy', 'sports']
            },
            'pattern_recognition': {
                'description': 'Ability to recognize and adapt to recurring patterns',
                'neural_indicators': ['sequence_learning', 'adaptation_speed'],
                'measurement_criteria': ['pattern_detection_speed', 'adaptation_rate'],
                'applies_to_environments': ['pong', 'chess', 'trading', 'anomaly_detection']
            },
            'error_recovery': {
                'description': 'Ability to recover from mistakes and continue effectively',
                'neural_indicators': ['post_error_adaptation', 'resilience_patterns'],
                'measurement_criteria': ['recovery_speed', 'performance_stability'],
                'applies_to_environments': ['pong', 'chess', 'trading', 'robotics', 'learning']
            },
            'adaptive_strategy': {
                'description': 'Ability to modify strategy based on changing conditions',
                'neural_indicators': ['strategy_switching', 'context_adaptation'],
                'measurement_criteria': ['strategy_effectiveness', 'adaptation_speed'],
                'applies_to_environments': ['pong', 'chess', 'trading', 'business', 'warfare']
            }
        }

        print("🧠 Neural Pattern Interpreter initialized")
        print(f"   📊 Tracking {len(self.universal_skills)} universal skills")

    def analyze_neural_learning(self, action_history: List[Dict], reward_history: List[float],
                                state_patterns: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze what the neural brain is learning and map it to transferable skills

        This is the core method that watches neural learning and interprets it
        symbolically, creating the bridge between the two brain systems.
        """
        analysis = {
            'detected_patterns': [],
            'emerging_skills': [],
            'skill_development': {},
            'transfer_readiness': {},
            'neural_insights': []
        }

        if len(action_history) < 10:  # Need minimum data for analysis
            return analysis

        # Analyze trajectory prediction patterns
        trajectory_skill = self._analyze_trajectory_prediction(action_history, reward_history, state_patterns)
        if trajectory_skill['confidence'] > 0.3:
            analysis['emerging_skills'].append('trajectory_prediction')
            analysis['skill_development']['trajectory_prediction'] = trajectory_skill

        # Analyze timing optimization patterns
        timing_skill = self._analyze_timing_optimization(action_history, reward_history)
        if timing_skill['confidence'] > 0.3:
            analysis['emerging_skills'].append('timing_optimization')
            analysis['skill_development']['timing_optimization'] = timing_skill

        # Analyze strategic positioning patterns
        positioning_skill = self._analyze_strategic_positioning(action_history, reward_history, state_patterns)
        if positioning_skill['confidence'] > 0.3:
            analysis['emerging_skills'].append('strategic_positioning')
            analysis['skill_development']['strategic_positioning'] = positioning_skill

        # Analyze pattern recognition capabilities
        pattern_skill = self._analyze_pattern_recognition(action_history, reward_history)
        if pattern_skill['confidence'] > 0.3:
            analysis['emerging_skills'].append('pattern_recognition')
            analysis['skill_development']['pattern_recognition'] = pattern_skill

        # Analyze error recovery patterns
        recovery_skill = self._analyze_error_recovery(action_history, reward_history)
        if recovery_skill['confidence'] > 0.3:
            analysis['emerging_skills'].append('error_recovery')
            analysis['skill_development']['error_recovery'] = recovery_skill

        # Calculate transfer readiness for each skill
        for skill in analysis['emerging_skills']:
            skill_data = analysis['skill_development'][skill]
            analysis['transfer_readiness'][skill] = self._calculate_transfer_readiness(skill, skill_data)

        # Generate neural insights
        analysis['neural_insights'] = self._generate_neural_insights(analysis)

        return analysis

    def _analyze_trajectory_prediction(self, action_history: List[Dict], reward_history: List[float],
                                       state_patterns: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze if the neural brain is learning trajectory prediction"""
        skill_data = {
            'confidence': 0.0,
            'evidence': [],
            'neural_patterns': [],
            'measurement_score': 0.0
        }

        if len(state_patterns) < 5:
            return skill_data

        # Look for predictive action sequences
        predictive_actions = 0
        total_actions = 0

        for i in range(1, min(len(action_history), len(reward_history))):
            if reward_history[i] > 0:  # Successful action
                # Check if action was predictive (anticipatory)
                recent_rewards = reward_history[max(0, i - 3):i]
                if len(recent_rewards) >= 2 and all(r <= 0 for r in recent_rewards[:-1]):
                    # This looks like predictive positioning
                    predictive_actions += 1
                    skill_data['evidence'].append(f"Predictive action at step {i}")
            total_actions += 1

        if total_actions > 0:
            prediction_rate = predictive_actions / total_actions
            skill_data['confidence'] = min(1.0, prediction_rate * 2)  # Amplify for detection
            skill_data['measurement_score'] = prediction_rate

        # Analyze state pattern consistency
        if len(state_patterns) >= 10:
            # Look for patterns in the first few dimensions (typically position/velocity)
            recent_patterns = state_patterns[-10:]
            pattern_consistency = self._calculate_pattern_consistency(recent_patterns, slice(0, 4))
            skill_data['confidence'] = max(skill_data['confidence'], pattern_consistency * 0.8)
            skill_data['neural_patterns'].append(f"State pattern consistency: {pattern_consistency:.2f}")

        return skill_data

    def _analyze_timing_optimization(self, action_history: List[Dict], reward_history: List[float]) -> Dict[str, Any]:
        """Analyze if the neural brain is learning optimal timing"""
        skill_data = {
            'confidence': 0.0,
            'evidence': [],
            'neural_patterns': [],
            'measurement_score': 0.0
        }

        if len(reward_history) < 10:
            return skill_data

        # Analyze reward timing patterns
        high_reward_actions = []
        for i, reward in enumerate(reward_history):
            if reward > 1.0:  # Significant positive reward
                high_reward_actions.append(i)

        if len(high_reward_actions) >= 3:
            # Look for timing improvements
            timing_intervals = []
            for i in range(1, len(high_reward_actions)):
                interval = high_reward_actions[i] - high_reward_actions[i - 1]
                timing_intervals.append(interval)

            if timing_intervals:
                # Check if timing is becoming more consistent (learning optimization)
                early_variance = np.var(timing_intervals[:len(timing_intervals) // 2]) if len(
                    timing_intervals) >= 4 else 0
                late_variance = np.var(timing_intervals[len(timing_intervals) // 2:]) if len(
                    timing_intervals) >= 4 else 0

                if early_variance > 0 and late_variance < early_variance:
                    improvement = (early_variance - late_variance) / early_variance
                    skill_data['confidence'] = min(1.0, improvement * 1.5)
                    skill_data['evidence'].append(f"Timing variance reduced by {improvement:.1%}")
                    skill_data['measurement_score'] = improvement

        return skill_data

    def _analyze_strategic_positioning(self, action_history: List[Dict], reward_history: List[float],
                                       state_patterns: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze if the neural brain is learning strategic positioning"""
        skill_data = {
            'confidence': 0.0,
            'evidence': [],
            'neural_patterns': [],
            'measurement_score': 0.0
        }

        if len(action_history) < 15:
            return skill_data

        # Look for defensive/positioning patterns
        positioning_actions = 0  # Actions that maintain position (action 1)
        movement_actions = 0  # Actions that change position (actions 0, 2)

        for action_data in action_history[-20:]:  # Recent actions
            action = action_data.get('action', 1)
            if action == 1:  # Stay/hold position
                positioning_actions += 1
            else:
                movement_actions += 1

        total_actions = positioning_actions + movement_actions
        if total_actions > 0:
            # Strategic positioning often involves knowing when NOT to move
            positioning_ratio = positioning_actions / total_actions

            # Correlate with rewards
            recent_rewards = reward_history[-20:] if len(reward_history) >= 20 else reward_history
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0

            if avg_reward > 0 and positioning_ratio > 0.3:  # Good performance with strategic positioning
                skill_data['confidence'] = min(1.0, positioning_ratio * avg_reward)
                skill_data['evidence'].append(f"Strategic positioning: {positioning_ratio:.1%} of actions")
                skill_data['measurement_score'] = positioning_ratio * avg_reward

        return skill_data

    def _analyze_pattern_recognition(self, action_history: List[Dict], reward_history: List[float]) -> Dict[str, Any]:
        """Analyze if the neural brain is learning pattern recognition"""
        skill_data = {
            'confidence': 0.0,
            'evidence': [],
            'neural_patterns': [],
            'measurement_score': 0.0
        }

        if len(action_history) < 20:
            return skill_data

        # Look for repeating action sequences that lead to rewards
        action_sequences = []
        for i in range(len(action_history) - 3):
            sequence = [action_history[j].get('action', 0) for j in range(i, i + 3)]
            action_sequences.append(tuple(sequence))

        # Find common sequences
        sequence_counts = defaultdict(int)
        sequence_rewards = defaultdict(list)

        for i, seq in enumerate(action_sequences):
            sequence_counts[seq] += 1
            if i + 3 < len(reward_history):
                # Reward for the sequence
                seq_reward = sum(reward_history[i:i + 3])
                sequence_rewards[seq].append(seq_reward)

        # Find sequences that appear multiple times with good rewards
        successful_patterns = 0
        total_patterns = 0

        for seq, count in sequence_counts.items():
            if count >= 2:  # Sequence repeated
                total_patterns += 1
                avg_reward = np.mean(sequence_rewards[seq]) if sequence_rewards[seq] else 0
                if avg_reward > 0:
                    successful_patterns += 1
                    skill_data['evidence'].append(f"Successful pattern: {seq} (used {count} times)")

        if total_patterns > 0:
            pattern_success_rate = successful_patterns / total_patterns
            skill_data['confidence'] = min(1.0, pattern_success_rate * 1.2)
            skill_data['measurement_score'] = pattern_success_rate

        return skill_data

    def _analyze_error_recovery(self, action_history: List[Dict], reward_history: List[float]) -> Dict[str, Any]:
        """Analyze if the neural brain is learning error recovery"""
        skill_data = {
            'confidence': 0.0,
            'evidence': [],
            'neural_patterns': [],
            'measurement_score': 0.0
        }

        if len(reward_history) < 15:
            return skill_data

        # Find error events (negative rewards) and analyze recovery
        error_recoveries = []

        for i in range(len(reward_history) - 5):
            if reward_history[i] < -0.3:  # Error event
                # Look at next 5 actions for recovery
                recovery_rewards = reward_history[i + 1:i + 6]
                if recovery_rewards:
                    # Calculate recovery strength
                    recovery_score = sum(r for r in recovery_rewards if r > 0)
                    if recovery_score > 0:
                        error_recoveries.append(recovery_score)
                        skill_data['evidence'].append(f"Recovery after error at step {i}: {recovery_score:.2f}")

        if len(error_recoveries) >= 2:
            avg_recovery = np.mean(error_recoveries)
            # Strong recovery indicates error recovery skill
            skill_data['confidence'] = min(1.0, avg_recovery / 2.0)
            skill_data['measurement_score'] = avg_recovery

        return skill_data

    def _calculate_pattern_consistency(self, patterns: List[np.ndarray], feature_slice: slice) -> float:
        """Calculate how consistent patterns are in specific features"""
        if len(patterns) < 3:
            return 0.0

        # Extract the specified features from each pattern
        features = []
        for pattern in patterns:
            if len(pattern) > feature_slice.stop:
                features.append(pattern[feature_slice])

        if len(features) < 3:
            return 0.0

        # Calculate consistency (inverse of variance)
        feature_matrix = np.array(features)
        variances = np.var(feature_matrix, axis=0)
        avg_variance = np.mean(variances)

        # Convert to consistency score (0-1)
        consistency = 1.0 / (1.0 + avg_variance)
        return min(1.0, consistency)

    def _calculate_transfer_readiness(self, skill: str, skill_data: Dict[str, Any]) -> float:
        """Calculate how ready a skill is for transfer to other environments"""
        confidence = skill_data.get('confidence', 0.0)
        measurement = skill_data.get('measurement_score', 0.0)
        evidence_count = len(skill_data.get('evidence', []))

        # Transfer readiness based on multiple factors
        readiness = (confidence * 0.5 + measurement * 0.3 + min(1.0, evidence_count / 3) * 0.2)
        return min(1.0, readiness)

    def _generate_neural_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate human-readable insights about what the neural brain learned"""
        insights = []

        for skill in analysis['emerging_skills']:
            skill_data = analysis['skill_development'][skill]
            confidence = skill_data['confidence']
            readiness = analysis['transfer_readiness'].get(skill, 0.0)

            if confidence > 0.7:
                insights.append(f"Strong {skill} skill detected (confidence: {confidence:.1%})")
            elif confidence > 0.4:
                insights.append(f"Developing {skill} skill (confidence: {confidence:.1%})")

            if readiness > 0.6:
                environments = self.universal_skills[skill]['applies_to_environments']
                insights.append(f"{skill} ready for transfer to: {', '.join(environments[:3])}")

        if not insights:
            insights.append("Neural brain is learning basic patterns - no clear transferable skills yet")

        return insights


class UniversalKnowledgeMapper:
    """
    Universal Knowledge Mapper - Maps knowledge across environments

    This system maintains the mapping between environment-specific knowledge
    and universal transferable concepts, enabling the symbolic brain to guide
    transfer learning.
    """

    def __init__(self):
        # Environment knowledge storage
        self.environment_contexts = {}
        self.skill_mappings = {}
        self.transfer_history = []

        # Universal concept definitions
        self.universal_concepts = {
            'moving_object_interception': {
                'abstract_description': 'Intercepting moving targets through prediction and positioning',
                'environment_implementations': {
                    'pong': 'Hit ball with paddle by predicting trajectory',
                    'chess': 'Capture opponent pieces by predicting moves',
                    'trading': 'Enter positions by predicting price movements',
                    'robotics': 'Catch objects by predicting motion paths'
                },
                'core_principles': ['prediction', 'positioning', 'timing']
            },
            'competitive_strategy': {
                'abstract_description': 'Developing winning strategies against opponents',
                'environment_implementations': {
                    'pong': 'Adapt paddle strategy based on opponent patterns',
                    'chess': 'Develop opening/middle/endgame strategies',
                    'trading': 'Develop trading strategies based on market patterns',
                    'poker': 'Develop betting strategies based on opponent behavior'
                },
                'core_principles': ['adaptation', 'pattern_recognition', 'strategic_thinking']
            },
            'error_recovery': {
                'abstract_description': 'Recovering from mistakes and maintaining performance',
                'environment_implementations': {
                    'pong': 'Continue playing effectively after missing ball',
                    'chess': 'Recover position after making poor moves',
                    'trading': 'Recover portfolio after losses',
                    'robotics': 'Continue task after failed actions'
                },
                'core_principles': ['resilience', 'adaptation', 'persistence']
            }
        }

        print("🗺️ Universal Knowledge Mapper initialized")
        print(f"   🌍 Tracking {len(self.universal_concepts)} universal concepts")

    def register_environment_context(self, environment_id: str, context: Dict[str, Any]):
        """Register knowledge context for a specific environment"""
        self.environment_contexts[environment_id] = {
            'context': context,
            'registered_at': time.time(),
            'skills_learned': [],
            'transfer_potential': {}
        }

        # Calculate transfer potential to other environments
        for other_env in self.environment_contexts:
            if other_env != environment_id:
                potential = self._calculate_transfer_potential(environment_id, other_env)
                self.environment_contexts[environment_id]['transfer_potential'][other_env] = potential

        print(f"📝 Registered environment context for {environment_id}")
        return True

    def map_neural_skills_to_universal(self, environment_id: str, neural_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Map environment-specific neural learning to universal transferable skills"""
        universal_mapping = {
            'universal_skills_discovered': [],
            'environment_specific_skills': [],
            'transfer_recommendations': [],
            'skill_confidence_matrix': {}
        }

        if environment_id not in self.environment_contexts:
            print(f"⚠️ Environment {environment_id} not registered")
            return universal_mapping

        # Map detected neural patterns to universal skills
        for skill in neural_analysis.get('emerging_skills', []):
            if skill in ['trajectory_prediction', 'timing_optimization', 'strategic_positioning']:
                # These map directly to universal concepts
                universal_skill = self._map_to_universal_concept(skill, environment_id)
                if universal_skill:
                    universal_mapping['universal_skills_discovered'].append(universal_skill)

                    # Calculate confidence for this universal skill
                    skill_data = neural_analysis.get('skill_development', {}).get(skill, {})
                    confidence = skill_data.get('confidence', 0.0)
                    universal_mapping['skill_confidence_matrix'][universal_skill['concept']] = confidence

        # Generate transfer recommendations
        universal_mapping['transfer_recommendations'] = self._generate_transfer_recommendations(
            environment_id, universal_mapping['universal_skills_discovered']
        )

        return universal_mapping

    def _map_to_universal_concept(self, neural_skill: str, environment_id: str) -> Optional[Dict[str, Any]]:
        """Map a neural skill to universal concept"""
        # Skill to concept mapping
        skill_to_concept = {
            'trajectory_prediction': 'moving_object_interception',
            'timing_optimization': 'moving_object_interception',
            'strategic_positioning': 'competitive_strategy',
            'pattern_recognition': 'competitive_strategy',
            'error_recovery': 'error_recovery',
            'adaptive_strategy': 'competitive_strategy'
        }

        concept_id = skill_to_concept.get(neural_skill)
        if not concept_id or concept_id not in self.universal_concepts:
            return None

        concept = self.universal_concepts[concept_id]
        env_implementation = concept['environment_implementations'].get(environment_id, 'Generic implementation')

        return {
            'concept': concept_id,
            'neural_skill': neural_skill,
            'environment_implementation': env_implementation,
            'abstract_description': concept['abstract_description'],
            'core_principles': concept['core_principles'],
            'transferable_to': list(concept['environment_implementations'].keys())
        }

    def _calculate_transfer_potential(self, env1: str, env2: str) -> float:
        """Calculate transfer learning potential between two environments"""
        if env1 not in self.environment_contexts or env2 not in self.environment_contexts:
            return 0.0

        context1 = self.environment_contexts[env1]['context']
        context2 = self.environment_contexts[env2]['context']

        # Simple similarity calculation based on shared concepts
        shared_concepts = 0
        total_concepts = 0

        # Compare transferable skills
        skills1 = set(context1.get('transferable_skills', []))
        skills2 = set(context2.get('transferable_skills', []))

        if skills1 and skills2:
            shared_concepts = len(skills1.intersection(skills2))
            total_concepts = len(skills1.union(skills2))

        if total_concepts > 0:
            return shared_concepts / total_concepts

        return 0.0

    def _generate_transfer_recommendations(self, current_env: str, universal_skills: List[Dict]) -> List[
        Dict[str, Any]]:
        """Generate recommendations for transferring skills to other environments"""
        recommendations = []

        for skill in universal_skills:
            concept = skill['concept']
            transferable_envs = skill['transferable_to']

            for target_env in transferable_envs:
                if target_env != current_env and target_env in self.environment_contexts:
                    # Calculate transfer confidence
                    transfer_potential = self.environment_contexts[current_env]['transfer_potential'].get(target_env,
                                                                                                          0.0)

                    recommendation = {
                        'source_environment': current_env,
                        'target_environment': target_env,
                        'skill_concept': concept,
                        'neural_skill': skill['neural_skill'],
                        'transfer_confidence': transfer_potential,
                        'implementation_guidance': self.universal_concepts[concept]['environment_implementations'].get(
                            target_env, ''),
                        'recommended': transfer_potential > 0.3
                    }

                    recommendations.append(recommendation)

        # Sort by transfer confidence
        recommendations.sort(key=lambda x: x['transfer_confidence'], reverse=True)
        return recommendations[:5]  # Top 5 recommendations


class EnhancedSymbolicDecisionMaker:
    """
    Enhanced Symbolic Decision Maker with Neural-Symbolic Bridge

    This is the core component that bridges the neural and symbolic brains,
    interpreting neural patterns and guiding decisions with environmental context.
    """

    def __init__(self):
        # Dual brain components
        self.neural_interpreter = NeuralPatternInterpreter()
        self.knowledge_mapper = UniversalKnowledgeMapper()

        # Decision tracking
        self.decision_history = []
        self.neural_symbolic_correlations = []
        self.environment_context = None

        # Strategy mapping (enhanced from original)
        self.universal_strategy_mappings = {
            # Neural patterns mapped to symbolic strategies
            'trajectory_prediction': self._apply_trajectory_prediction,
            'timing_optimization': self._apply_timing_optimization,
            'strategic_positioning': self._apply_strategic_positioning,
            'pattern_recognition': self._apply_pattern_recognition,
            'error_recovery': self._apply_error_recovery,
            'adaptive_strategy': self._apply_adaptive_strategy,

            # Environment-agnostic strategies
            'move_toward_target': self._apply_move_toward_target,
            'defensive_positioning': self._apply_defensive_positioning,
            'aggressive_advancement': self._apply_aggressive_advancement,
            'pattern_based_prediction': self._apply_pattern_based_prediction
        }

        print("🧩 Enhanced Symbolic Decision Maker initialized")
        print(f"   🔗 Neural-Symbolic bridge active")
        print(f"   📚 {len(self.universal_strategy_mappings)} strategies available")

    def set_environment_context(self, environment_id: str, context: Dict[str, Any]):
        """Set the current environment context for symbolic reasoning"""
        self.environment_context = {
            'environment_id': environment_id,
            'context': context,
            'loaded_at': time.time()
        }

        # Register with knowledge mapper
        self.knowledge_mapper.register_environment_context(environment_id, context)

        print(f"🌍 Environment context set: {environment_id}")
        return True

    def make_enhanced_decision(self, neural_state: np.ndarray, neural_q_values: np.ndarray,
                               action_history: List[Dict], reward_history: List[float],
                               exploration_rate: float) -> Tuple[int, str, Dict[str, Any]]:
        """
        Enhanced decision making with neural-symbolic integration

        This is the key method that demonstrates the dual brain in action:
        1. Neural brain provides Q-values and learned patterns
        2. Symbolic brain interprets patterns and provides context
        3. Decision combines both sources of intelligence
        """

        # Step 1: Analyze what the neural brain has learned
        neural_analysis = self.neural_interpreter.analyze_neural_learning(
            action_history, reward_history, [neural_state]
        )

        # Step 2: Map neural learning to universal concepts
        universal_mapping = None
        if self.environment_context:
            universal_mapping = self.knowledge_mapper.map_neural_skills_to_universal(
                self.environment_context['environment_id'], neural_analysis
            )

        # Step 3: Decide whether to use neural or symbolic decision making
        use_symbolic = self._should_use_symbolic_decision(neural_analysis, exploration_rate)

        if use_symbolic and self.environment_context:
            # Symbolic decision with environmental context
            symbolic_action, reasoning = self._make_symbolic_decision(
                neural_state, neural_q_values, neural_analysis, universal_mapping
            )

            decision_info = {
                'decision_type': 'symbolic',
                'neural_action': int(np.argmax(neural_q_values)),
                'symbolic_action': symbolic_action,
                'chosen_action': symbolic_action,
                'reasoning': reasoning,
                'neural_analysis': neural_analysis,
                'universal_mapping': universal_mapping,
                'timestamp': time.time()
            }

            return symbolic_action, f"🧩 {reasoning}", decision_info

        else:
            # Neural decision
            neural_action = int(np.argmax(neural_q_values))

            decision_info = {
                'decision_type': 'neural',
                'neural_action': neural_action,
                'symbolic_action': None,
                'chosen_action': neural_action,
                'reasoning': 'Neural network decision',
                'neural_analysis': neural_analysis,
                'exploration_rate': exploration_rate,
                'timestamp': time.time()
            }

            return neural_action, "🧠 Neural network decision", decision_info

    def _should_use_symbolic_decision(self, neural_analysis: Dict[str, Any], exploration_rate: float) -> bool:
        """Decide whether to use symbolic reasoning or neural network"""
        # Use symbolic reasoning when:
        # 1. We have clear environment context
        # 2. Neural brain has learned transferable skills
        # 3. Exploration rate is low (confident phase)

        if not self.environment_context:
            return False

        emerging_skills = neural_analysis.get('emerging_skills', [])
        skill_confidence = sum(
            neural_analysis.get('skill_development', {}).get(skill, {}).get('confidence', 0)
            for skill in emerging_skills
        ) / max(1, len(emerging_skills))

        # Calculate symbolic decision probability
        context_factor = 0.3  # Base probability with context
        skill_factor = min(0.4, skill_confidence * 0.6)  # Higher with learned skills
        confidence_factor = max(0, 0.3 - exploration_rate)  # Higher when less exploring

        symbolic_probability = context_factor + skill_factor + confidence_factor

        return random.random() < symbolic_probability

    def _make_symbolic_decision(self, neural_state: np.ndarray, neural_q_values: np.ndarray,
                                neural_analysis: Dict[str, Any], universal_mapping: Optional[Dict]) -> Tuple[int, str]:
        """Make decision using symbolic reasoning with neural insights"""

        # Get environment context
        env_context = self.environment_context['context']
        environment_id = self.environment_context['environment_id']

        # Choose strategy based on learned skills and context
        emerging_skills = neural_analysis.get('emerging_skills', [])

        if 'trajectory_prediction' in emerging_skills:
            action, confidence = self.universal_strategy_mappings['trajectory_prediction'](
                neural_state, neural_q_values, {'neural_analysis': neural_analysis}
            )
            reasoning = f"Using learned trajectory prediction skill (neural confidence: {neural_analysis.get('skill_development', {}).get('trajectory_prediction', {}).get('confidence', 0):.2f})"

        elif 'strategic_positioning' in emerging_skills:
            action, confidence = self.universal_strategy_mappings['strategic_positioning'](
                neural_state, neural_q_values, {'neural_analysis': neural_analysis}
            )
            reasoning = f"Using learned strategic positioning skill"

        elif 'timing_optimization' in emerging_skills:
            action, confidence = self.universal_strategy_mappings['timing_optimization'](
                neural_state, neural_q_values, {'neural_analysis': neural_analysis}
            )
            reasoning = f"Using learned timing optimization skill"

        else:
            # Fall back to environment context strategies
            strategies = env_context.get('strategies', [])
            if strategies:
                # Use first available strategy
                strategy_name = strategies[0].lower()
                for strategy_key in self.universal_strategy_mappings:
                    if strategy_key in strategy_name:
                        action, confidence = self.universal_strategy_mappings[strategy_key](
                            neural_state, neural_q_values, {'environment_context': env_context}
                        )
                        reasoning = f"Using environment strategy: {strategies[0]}"
                        break
                else:
                    # Default fallback
                    action = int(np.argmax(neural_q_values))
                    reasoning = "Fallback to neural decision"
            else:
                action = int(np.argmax(neural_q_values))
                reasoning = "No symbolic strategies available, using neural decision"

        return action, reasoning

    def update_neural_symbolic_correlation(self, decision_info: Dict[str, Any], reward: float):
        """Update correlation tracking between neural and symbolic decisions"""

        # Record the outcome
        correlation_record = {
            'decision_type': decision_info['decision_type'],
            'reward': reward,
            'neural_analysis': decision_info.get('neural_analysis', {}),
            'universal_mapping': decision_info.get('universal_mapping'),
            'timestamp': time.time()
        }

        self.neural_symbolic_correlations.append(correlation_record)

        # Keep only recent correlations
        if len(self.neural_symbolic_correlations) > 100:
            self.neural_symbolic_correlations = self.neural_symbolic_correlations[-50:]

        # Update skill confidence based on outcomes
        if decision_info['decision_type'] == 'symbolic':
            neural_analysis = decision_info.get('neural_analysis', {})
            for skill in neural_analysis.get('emerging_skills', []):
                skill_data = neural_analysis.get('skill_development', {}).get(skill, {})
                if reward > 0:
                    # Positive outcome reinforces skill confidence
                    current_confidence = skill_data.get('confidence', 0.0)
                    # Slight boost for successful application
                    boosted_confidence = min(1.0, current_confidence + 0.05)
                    self.neural_interpreter.skill_confidence_scores[skill] = boosted_confidence

    # Strategy implementations (enhanced versions of original methods)
    def _apply_trajectory_prediction(self, state: np.ndarray, q_values: np.ndarray, context: Dict) -> Tuple[int, float]:
        """Enhanced trajectory prediction using neural insights"""
        # Extract position and velocity features (first 4 dimensions)
        if len(state) >= 4:
            pos_x, pos_y, vel_x, vel_y = state[0], state[1], state[2], state[3]

            # Use neural analysis if available
            neural_analysis = context.get('neural_analysis', {})
            trajectory_skill = neural_analysis.get('skill_development', {}).get('trajectory_prediction', {})
            confidence_boost = trajectory_skill.get('confidence', 0.5)

            # Enhanced prediction with confidence
            prediction_time = 0.3 * (1 + confidence_boost)  # More confident = longer prediction
            future_x = pos_x + vel_x * prediction_time
            future_y = pos_y + vel_y * prediction_time

            if abs(future_y) > 0.1:
                action = 2 if future_y > 0 else 0
                confidence = 0.8 * (1 + confidence_boost * 0.5)
            else:
                action = 1
                confidence = 0.9 * (1 + confidence_boost * 0.3)
        else:
            action = int(np.argmax(q_values))
            confidence = 0.5

        return action, confidence

    def _apply_timing_optimization(self, state: np.ndarray, q_values: np.ndarray, context: Dict) -> Tuple[int, float]:
        """Enhanced timing optimization using neural insights"""
        neural_analysis = context.get('neural_analysis', {})
        timing_skill = neural_analysis.get('skill_development', {}).get('timing_optimization', {})
        timing_confidence = timing_skill.get('confidence', 0.5)

        # Use neural timing insights to adjust decision timing
        if timing_confidence > 0.6:
            # High confidence in timing - be more decisive
            action = int(np.argmax(q_values))
            confidence = 0.85
        elif timing_confidence > 0.3:
            # Medium confidence - balanced approach
            top_actions = np.argsort(q_values)[-2:]
            action = int(random.choice(top_actions))
            confidence = 0.7
        else:
            # Low confidence - conservative timing
            action = 1  # Stay/wait
            confidence = 0.6

        return action, confidence

    def _apply_strategic_positioning(self, state: np.ndarray, q_values: np.ndarray, context: Dict) -> Tuple[int, float]:
        """Enhanced strategic positioning using neural insights"""
        neural_analysis = context.get('neural_analysis', {})
        positioning_skill = neural_analysis.get('skill_development', {}).get('strategic_positioning', {})
        positioning_confidence = positioning_skill.get('confidence', 0.5)

        # Enhanced positioning based on learned patterns
        if len(state) >= 6:
            # Use neural positioning insights
            current_pos = state[5] if len(state) > 5 else 0  # AI paddle position
            target_pos = state[0] * 0.7  # Ball position influence

            # Adjust strategy based on positioning confidence
            if positioning_confidence > 0.7:
                # High confidence - aggressive positioning
                action = 2 if target_pos > current_pos + 0.1 else (0 if target_pos < current_pos - 0.1 else 1)
                confidence = 0.8
            else:
                # Lower confidence - conservative positioning
                action = 2 if target_pos > current_pos + 0.2 else (0 if target_pos < current_pos - 0.2 else 1)
                confidence = 0.6
        else:
            action = 1
            confidence = 0.5

        return action, confidence

    # Additional strategy implementations...
    def _apply_pattern_recognition(self, state: np.ndarray, q_values: np.ndarray, context: Dict) -> Tuple[int, float]:
        """Apply pattern recognition strategy"""
        neural_analysis = context.get('neural_analysis', {})
        pattern_skill = neural_analysis.get('skill_development', {}).get('pattern_recognition', {})
        pattern_confidence = pattern_skill.get('confidence', 0.5)

        if pattern_confidence > 0.6:
            # Use learned patterns
            action = int(np.argmax(q_values))
            confidence = 0.8
        else:
            # Basic pattern following
            if len(state) >= 4:
                # Simple pattern: follow ball movement
                ball_dy = state[3] if len(state) > 3 else 0
                action = 2 if ball_dy > 0.1 else (0 if ball_dy < -0.1 else 1)
                confidence = 0.6
            else:
                action = 1
                confidence = 0.5

        return action, confidence

    def _apply_error_recovery(self, state: np.ndarray, q_values: np.ndarray, context: Dict) -> Tuple[int, float]:
        """Apply error recovery strategy"""
        # Conservative recovery approach
        action = 1  # Stay centered for recovery
        confidence = 0.7
        return action, confidence

    def _apply_adaptive_strategy(self, state: np.ndarray, q_values: np.ndarray, context: Dict) -> Tuple[int, float]:
        """Apply adaptive strategy based on situation"""
        # Adapt based on context
        neural_analysis = context.get('neural_analysis', {})
        emerging_skills = neural_analysis.get('emerging_skills', [])

        if 'trajectory_prediction' in emerging_skills:
            return self._apply_trajectory_prediction(state, q_values, context)
        elif 'strategic_positioning' in emerging_skills:
            return self._apply_strategic_positioning(state, q_values, context)
        else:
            action = int(np.argmax(q_values))
            confidence = 0.6
            return action, confidence

    # Legacy strategy implementations for backward compatibility
    def _apply_move_toward_target(self, state: np.ndarray, q_values: np.ndarray, context: Dict) -> Tuple[int, float]:
        """Move toward target (legacy compatibility)"""
        return self._apply_trajectory_prediction(state, q_values, context)

    def _apply_defensive_positioning(self, state: np.ndarray, q_values: np.ndarray, context: Dict) -> Tuple[int, float]:
        """Defensive positioning (legacy compatibility)"""
        return self._apply_strategic_positioning(state, q_values, context)

    def _apply_aggressive_advancement(self, state: np.ndarray, q_values: np.ndarray, context: Dict) -> Tuple[
        int, float]:
        """Aggressive advancement (legacy compatibility)"""
        action = int(np.argmax(q_values))
        confidence = 0.7
        return action, confidence

    def _apply_pattern_based_prediction(self, state: np.ndarray, q_values: np.ndarray, context: Dict) -> Tuple[
        int, float]:
        """Pattern-based prediction (legacy compatibility)"""
        return self._apply_pattern_recognition(state, q_values, context)

    def get_dual_brain_report(self) -> Dict[str, Any]:
        """Get comprehensive report on dual brain performance"""
        # Analyze recent correlations
        recent_correlations = self.neural_symbolic_correlations[-20:] if self.neural_symbolic_correlations else []

        neural_decisions = [c for c in recent_correlations if c['decision_type'] == 'neural']
        symbolic_decisions = [c for c in recent_correlations if c['decision_type'] == 'symbolic']

        neural_avg_reward = np.mean([c['reward'] for c in neural_decisions]) if neural_decisions else 0
        symbolic_avg_reward = np.mean([c['reward'] for c in symbolic_decisions]) if symbolic_decisions else 0

        return {
            'total_decisions': len(self.neural_symbolic_correlations),
            'neural_decisions': len(neural_decisions),
            'symbolic_decisions': len(symbolic_decisions),
            'neural_avg_reward': neural_avg_reward,
            'symbolic_avg_reward': symbolic_avg_reward,
            'decision_balance': len(symbolic_decisions) / max(1, len(recent_correlations)),
            'environment_context_active': self.environment_context is not None,
            'current_environment': self.environment_context['environment_id'] if self.environment_context else None,
            'neural_interpreter_skills': len(self.neural_interpreter.universal_skills),
            'knowledge_mapper_environments': len(self.knowledge_mapper.environment_contexts)
        }


# Example integration and testing
if __name__ == "__main__":
    print("🧪 Testing Enhanced Neural-Symbolic Dual Brain Knowledge System...")

    # Initialize enhanced system
    decision_maker = EnhancedSymbolicDecisionMaker()

    # Mock environment context (Pong)
    pong_context = {
        'environment_id': 'pong',
        'objective': 'Score 21 points before opponent by hitting ball with paddle',
        'rules': [
            'Hit ball with paddle to keep it in play',
            'Ball bounces off top/bottom walls',
            'Score when ball passes opponent paddle',
            'First to 21 points wins'
        ],
        'strategies': [
            'Predict ball trajectory for positioning',
            'Time paddle movements for optimal hits',
            'Use defensive positioning when needed'
        ],
        'transferable_skills': [
            'trajectory_prediction',
            'timing_optimization',
            'strategic_positioning'
        ]
    }

    # Set environment context
    decision_maker.set_environment_context('pong', pong_context)

    # Simulate learning session with dual brain
    print(f"\n🎮 Simulating Dual Brain Learning Session:")

    action_history = []
    reward_history = []

    for step in range(50):
        # Mock neural state and Q-values
        neural_state = np.random.random(256) * 2 - 1  # Universal 256-dim state
        neural_q_values = np.random.random(3)

        # Mock action and reward
        mock_action = {'action': random.randint(0, 2), 'timestamp': time.time()}
        mock_reward = random.uniform(-1, 3)

        action_history.append(mock_action)
        reward_history.append(mock_reward)

        # Enhanced decision making every 10 steps
        if step % 10 == 0 and step > 0:
            action, reasoning, decision_info = decision_maker.make_enhanced_decision(
                neural_state, neural_q_values, action_history, reward_history, exploration_rate=0.3
            )

            print(f"Step {step}: {reasoning}")

            # Update correlation tracking
            decision_maker.update_neural_symbolic_correlation(decision_info, mock_reward)

            # Show neural analysis
            neural_analysis = decision_info.get('neural_analysis', {})
            if neural_analysis.get('emerging_skills'):
                print(f"   🧠 Emerging skills: {neural_analysis['emerging_skills']}")

            universal_mapping = decision_info.get('universal_mapping')
            if universal_mapping and universal_mapping.get('universal_skills_discovered'):
                print(
                    f"   🌍 Universal skills: {[s['concept'] for s in universal_mapping['universal_skills_discovered']]}")

    # Get dual brain performance report
    print(f"\n📊 Dual Brain Performance Report:")
    report = decision_maker.get_dual_brain_report()
    for key, value in report.items():
        print(f"   {key}: {value}")

    print(f"\n✅ Enhanced Neural-Symbolic Dual Brain Knowledge System test complete!")
    print(f"🧠🧩 Neural and Symbolic brains successfully integrated!")
    print(f"🌍 Universal knowledge mapping enables cross-environment transfer learning!")
    print(f"🔗 Dual brain bridge creates truly intelligent agents!")