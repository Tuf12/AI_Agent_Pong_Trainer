# dual_brain_system.py - Enhanced Dual Brain Architecture with Neural-Symbolic Integration
import json
import time
import os
import random
from datetime import datetime
from collections import defaultdict, deque
from copy import deepcopy
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Import the enhanced knowledge system for neural-symbolic integration
from knowledge_system import EnhancedSymbolicDecisionMaker, NeuralPatternInterpreter, UniversalKnowledgeMapper


class EnhancedAgentBrain:
    """
    Enhanced Core Learning Engine with Neural-Symbolic Integration

    This brain now connects with the symbolic brain to interpret what it's learning
    and extract transferable knowledge across environments.
    """

    def __init__(self, agent_id: str, environment_id: str, brain_file: str):
        self.agent_id = agent_id
        self.environment_id = environment_id
        self.brain_file = brain_file

        # Core learning metrics (transferable across environments)
        self.total_training_steps = 0
        self.total_target_updates = 0
        self.total_experience_gained = 0
        self.core_learning_efficiency = 0.0

        # Environment-specific learning metrics
        self.env_training_steps = 0
        self.env_learning_rate = 0.001
        self.env_gamma = 0.99
        self.env_epsilon = 0.8
        self.env_double_dqn_improvements = 0
        self.env_total_loss = 0.0

        # Enhanced neural-symbolic integration
        self.neural_pattern_interpreter = NeuralPatternInterpreter()
        self.neural_learning_history = []
        self.action_reward_history = []
        self.state_pattern_history = []

        # Transfer learning metrics
        self.transfer_learning_enabled = True
        self.knowledge_transfer_events = []
        self.cross_environment_correlations = {}
        self.transferable_patterns = []
        self.neural_skill_development = {}

        # Meta-learning principles (learned across all environments)
        self.meta_learning_principles = {
            'adaptation_speed': 0.0,
            'pattern_recognition_ability': 0.0,
            'error_recovery_rate': 0.0,
            'strategy_flexibility': 0.0,
            'learning_transfer_success': 0.0,
            'neural_symbolic_coherence': 0.0  # NEW: How well neural and symbolic agree
        }

        # Environment adaptation tracking
        self.environment_adaptations = {}
        self.environment_performance_history = []

        # Architecture metadata
        self.architecture_version = "Agent Byte v2.0 - Neural-Symbolic Integration"
        self.created_timestamp = time.time()
        self.last_updated = time.time()

        # Load existing brain data
        self.load_brain()

        print(f"🧠 Enhanced Agent Brain initialized with Neural-Symbolic Integration")

    def load_brain(self):
        """Load brain data with enhanced neural-symbolic integration"""
        try:
            if os.path.exists(self.brain_file):
                with open(self.brain_file, 'r') as f:
                    brain_data = json.load(f)

                # Load core metrics
                self.total_training_steps = brain_data.get('total_training_steps', 0)
                self.total_target_updates = brain_data.get('total_target_updates', 0)
                self.core_learning_efficiency = brain_data.get('core_learning_efficiency', 0.0)

                # Load environment-specific metrics
                self.env_training_steps = brain_data.get('training_steps', 0)
                self.env_learning_rate = brain_data.get('learning_rate', 0.001)
                self.env_gamma = brain_data.get('gamma', 0.99)
                self.env_epsilon = brain_data.get('epsilon', 0.8)
                self.env_total_loss = brain_data.get('total_loss', 0.0)

                # Load enhanced neural-symbolic data
                self.meta_learning_principles = brain_data.get('meta_learning_principles',
                                                               self.meta_learning_principles)
                self.environment_adaptations = brain_data.get('environment_adaptations', {})
                self.knowledge_transfer_events = brain_data.get('knowledge_transfer_events', [])
                self.neural_skill_development = brain_data.get('neural_skill_development', {})

                # Load neural learning history for pattern analysis
                self.neural_learning_history = brain_data.get('neural_learning_history', [])[-100:]  # Keep recent
                self.action_reward_history = brain_data.get('action_reward_history', [])[-200:]

                # Update timestamps
                self.created_timestamp = brain_data.get('created_timestamp', time.time())
                self.last_updated = time.time()

                print(
                    f"🧠 Enhanced brain loaded: {self.env_training_steps} steps, {len(self.neural_learning_history)} neural patterns")
                return True
            else:
                print(f"🆕 No existing enhanced brain found, starting fresh with neural-symbolic integration")
                return False

        except Exception as e:
            print(f"❌ Error loading enhanced brain: {e}")
            return False

    def save_brain(self):
        """Save enhanced brain data with neural-symbolic integration"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.brain_file), exist_ok=True)

            brain_data = {
                # Core transferable metrics
                'total_training_steps': self.total_training_steps,
                'total_target_updates': self.total_target_updates,
                'total_experience_gained': self.total_experience_gained,
                'core_learning_efficiency': self.core_learning_efficiency,

                # Environment-specific metrics
                'training_steps': self.env_training_steps,
                'learning_rate': self.env_learning_rate,
                'gamma': self.env_gamma,
                'epsilon': self.env_epsilon,
                'double_dqn_improvements': self.env_double_dqn_improvements,
                'target_updates': self.total_target_updates,
                'total_loss': self.env_total_loss,

                # Enhanced neural-symbolic integration
                'neural_learning_history': self.neural_learning_history[-100:],  # Keep recent patterns
                'action_reward_history': self.action_reward_history[-200:],
                'neural_skill_development': self.neural_skill_development,

                # Transfer learning data
                'transfer_learning_enabled': self.transfer_learning_enabled,
                'meta_learning_principles': self.meta_learning_principles,
                'environment_adaptations': self.environment_adaptations,
                'knowledge_transfer_events': self.knowledge_transfer_events[-50:],  # Keep recent events
                'cross_environment_correlations': self.cross_environment_correlations,

                # Metadata
                'agent_id': self.agent_id,
                'environment_id': self.environment_id,
                'architecture_version': self.architecture_version,
                'created_timestamp': self.created_timestamp,
                'last_updated': time.time(),
                'saved_at': datetime.now().isoformat()
            }

            with open(self.brain_file, 'w') as f:
                json.dump(brain_data, f, indent=2)

            print(f"🧠 Enhanced brain saved: {self.env_training_steps} steps, neural-symbolic integration active")
            return True

        except Exception as e:
            print(f"❌ Error saving enhanced brain: {e}")
            return False

    def record_enhanced_learning_step(self, reward: float, loss: float, exploration_rate: float,
                                      neural_state: np.ndarray, action: int):
        """Enhanced learning step recording with neural-symbolic analysis"""

        # Record basic learning step
        self.env_training_steps += 1
        self.total_training_steps += 1
        self.env_total_loss += loss
        self.env_epsilon = exploration_rate

        # Record neural learning pattern for symbolic interpretation
        neural_record = {
            'timestamp': time.time(),
            'reward': reward,
            'loss': loss,
            'exploration_rate': exploration_rate,
            'action': action,
            'state_features': neural_state[:10].tolist() if len(neural_state) >= 10 else neural_state.tolist(),
            'environment': self.environment_id
        }
        self.neural_learning_history.append(neural_record)

        # Record action-reward history for pattern analysis
        action_record = {
            'action': action,
            'reward': reward,
            'timestamp': time.time()
        }
        self.action_reward_history.append(action_record)

        # Keep histories manageable
        if len(self.neural_learning_history) > 100:
            self.neural_learning_history = self.neural_learning_history[-100:]
        if len(self.action_reward_history) > 200:
            self.action_reward_history = self.action_reward_history[-200:]

        # Analyze neural patterns every 10 steps
        if self.env_training_steps % 10 == 0:
            self._analyze_neural_patterns()

        # Update meta-learning principles
        self._update_enhanced_meta_learning_principles(reward, loss)

        # Record for cross-environment analysis
        self.environment_performance_history.append({
            'timestamp': time.time(),
            'reward': reward,
            'loss': loss,
            'exploration_rate': exploration_rate,
            'environment': self.environment_id,
            'neural_patterns_detected': len(self.neural_skill_development)
        })

        # Keep only recent history
        if len(self.environment_performance_history) > 1000:
            self.environment_performance_history = self.environment_performance_history[-1000:]

    def _analyze_neural_patterns(self):
        """Analyze neural learning patterns and extract transferable skills"""
        try:
            if len(self.action_reward_history) < 20:
                return

            # Extract action and reward histories for analysis
            action_history = [{'action': r['action'], 'timestamp': r['timestamp']} for r in self.action_reward_history]
            reward_history = [r['reward'] for r in self.action_reward_history]
            state_patterns = [np.array(r['state_features']) for r in self.neural_learning_history[-20:]]

            # Use neural pattern interpreter to analyze what's being learned
            neural_analysis = self.neural_pattern_interpreter.analyze_neural_learning(
                action_history, reward_history, state_patterns
            )

            # Update neural skill development tracking
            for skill in neural_analysis.get('emerging_skills', []):
                skill_data = neural_analysis.get('skill_development', {}).get(skill, {})

                if skill not in self.neural_skill_development:
                    self.neural_skill_development[skill] = {
                        'first_detected': time.time(),
                        'confidence_history': [],
                        'best_confidence': 0.0
                    }

                current_confidence = skill_data.get('confidence', 0.0)
                self.neural_skill_development[skill]['confidence_history'].append(current_confidence)
                self.neural_skill_development[skill]['best_confidence'] = max(
                    self.neural_skill_development[skill]['best_confidence'], current_confidence
                )

                # Keep confidence history manageable
                if len(self.neural_skill_development[skill]['confidence_history']) > 50:
                    self.neural_skill_development[skill]['confidence_history'] = \
                        self.neural_skill_development[skill]['confidence_history'][-25:]

            # Store neural insights for symbolic brain
            if neural_analysis.get('neural_insights'):
                self.transferable_patterns.extend(neural_analysis['neural_insights'])
                if len(self.transferable_patterns) > 20:
                    self.transferable_patterns = self.transferable_patterns[-20:]

            print(f"🔍 Neural pattern analysis: {len(neural_analysis.get('emerging_skills', []))} skills detected")

        except Exception as e:
            print(f"❌ Error analyzing neural patterns: {e}")

    def _update_enhanced_meta_learning_principles(self, reward: float, loss: float):
        """Update meta-learning principles with neural-symbolic coherence"""
        try:
            # Original meta-learning updates
            recent_losses = [h['loss'] for h in self.environment_performance_history[-10:]]
            if len(recent_losses) >= 10:
                loss_trend = np.mean(recent_losses[:5]) - np.mean(recent_losses[5:])
                self.meta_learning_principles['adaptation_speed'] = max(0, min(1, loss_trend))

            recent_rewards = [h['reward'] for h in self.environment_performance_history[-20:]]
            if len(recent_rewards) >= 20:
                reward_variance = np.var(recent_rewards)
                reward_consistency = max(0, 1 - (reward_variance / 10))
                self.meta_learning_principles['pattern_recognition_ability'] = reward_consistency

            if reward < 0:
                next_few_rewards = [h['reward'] for h in self.environment_performance_history[-5:]]
                if next_few_rewards:
                    recovery_rate = sum(1 for r in next_few_rewards if r > 0) / len(next_few_rewards)
                    current_recovery = self.meta_learning_principles['error_recovery_rate']
                    self.meta_learning_principles['error_recovery_rate'] = (current_recovery + recovery_rate) / 2

            # NEW: Neural-symbolic coherence measurement
            if len(self.neural_skill_development) > 0:
                # Measure how consistently neural patterns are developing
                skill_confidences = []
                for skill_data in self.neural_skill_development.values():
                    if skill_data['confidence_history']:
                        recent_confidence = np.mean(skill_data['confidence_history'][-5:])
                        skill_confidences.append(recent_confidence)

                if skill_confidences:
                    avg_skill_confidence = np.mean(skill_confidences)
                    self.meta_learning_principles['neural_symbolic_coherence'] = avg_skill_confidence

            # Update core learning efficiency
            if self.total_training_steps > 0:
                avg_reward = np.mean([h['reward'] for h in self.environment_performance_history[-100:]])
                self.core_learning_efficiency = max(0, min(1, (avg_reward + 5) / 10))

        except Exception as e:
            print(f"❌ Error updating enhanced meta-learning principles: {e}")

    def get_neural_insights_for_symbolic_brain(self) -> Dict[str, Any]:
        """Get neural learning insights for symbolic brain integration"""
        return {
            'neural_skill_development': self.neural_skill_development.copy(),
            'recent_neural_patterns': self.neural_learning_history[-10:],
            'action_reward_correlations': self._calculate_action_reward_correlations(),
            'learning_trajectory': self._calculate_learning_trajectory(),
            'transfer_readiness': self._calculate_neural_transfer_readiness(),
            'meta_learning_state': self.meta_learning_principles.copy()
        }

    def _calculate_action_reward_correlations(self) -> Dict[int, float]:
        """Calculate correlations between actions and rewards"""
        correlations = {}
        if len(self.action_reward_history) < 10:
            return correlations

        for action in [0, 1, 2]:  # Assume 3 actions
            action_rewards = [r['reward'] for r in self.action_reward_history if r['action'] == action]
            if action_rewards:
                correlations[action] = np.mean(action_rewards)
            else:
                correlations[action] = 0.0

        return correlations

    def _calculate_learning_trajectory(self) -> Dict[str, float]:
        """Calculate learning trajectory metrics"""
        if len(self.environment_performance_history) < 20:
            return {'trend': 0.0, 'stability': 0.0, 'acceleration': 0.0}

        recent_rewards = [h['reward'] for h in self.environment_performance_history[-20:]]
        early_rewards = recent_rewards[:10]
        late_rewards = recent_rewards[10:]

        trend = np.mean(late_rewards) - np.mean(early_rewards)
        stability = 1.0 - np.var(recent_rewards) / max(1, np.mean(np.abs(recent_rewards)))

        # Calculate acceleration (second derivative)
        if len(recent_rewards) >= 3:
            differences = np.diff(recent_rewards)
            acceleration = np.mean(np.diff(differences)) if len(differences) >= 2 else 0.0
        else:
            acceleration = 0.0

        return {
            'trend': trend,
            'stability': max(0, min(1, stability)),
            'acceleration': acceleration
        }

    def _calculate_neural_transfer_readiness(self) -> float:
        """Calculate how ready neural patterns are for transfer"""
        if not self.neural_skill_development:
            return 0.0

        readiness_scores = []
        for skill_data in self.neural_skill_development.values():
            if skill_data['confidence_history']:
                # Skill readiness based on consistency and peak confidence
                recent_confidence = np.mean(skill_data['confidence_history'][-5:])
                peak_confidence = skill_data['best_confidence']
                consistency = 1.0 - np.var(skill_data['confidence_history'][-10:]) if len(
                    skill_data['confidence_history']) >= 10 else 0.5

                skill_readiness = (recent_confidence * 0.4 + peak_confidence * 0.4 + consistency * 0.2)
                readiness_scores.append(skill_readiness)

        return np.mean(readiness_scores) if readiness_scores else 0.0

    def record_symbolic_feedback(self, decision_type: str, success_rate: float, skill_used: Optional[str] = None):
        """Record feedback from symbolic brain decisions"""
        feedback_record = {
            'timestamp': time.time(),
            'decision_type': decision_type,
            'success_rate': success_rate,
            'skill_used': skill_used,
            'environment': self.environment_id
        }

        if not hasattr(self, 'symbolic_feedback_history'):
            self.symbolic_feedback_history = []

        self.symbolic_feedback_history.append(feedback_record)

        # Update neural-symbolic coherence
        if len(self.symbolic_feedback_history) >= 10:
            recent_success = np.mean([f['success_rate'] for f in self.symbolic_feedback_history[-10:]])
            self.meta_learning_principles['neural_symbolic_coherence'] = recent_success

        # Keep history manageable
        if len(self.symbolic_feedback_history) > 50:
            self.symbolic_feedback_history = self.symbolic_feedback_history[-25:]


class EnhancedAgentKnowledge:
    """Enhanced Symbolic Knowledge with Persistent Environment Understanding"""

    def __init__(self, agent_id: str, environment_id: str, knowledge_file: str,
                 canonical_knowledge_file: Optional[str] = None,
                 default_knowledge_file: Optional[str] = None):
        self.agent_id = agent_id
        self.environment_id = environment_id
        self.knowledge_file = knowledge_file
        self.canonical_knowledge_file = canonical_knowledge_file or knowledge_file
        self.default_knowledge_file = default_knowledge_file
        self.canonical_knowledge = self._create_canonical_template()
        self.default_template = None

        # Initialize enhanced symbolic decision maker with neural integration
        self.symbolic_decision_maker = EnhancedSymbolicDecisionMaker()
        self.knowledge_mapper = UniversalKnowledgeMapper()

        # Enhanced multi-environment knowledge structure with environment profiles
        self.knowledge = {
            "general_knowledge": self._create_general_knowledge_section(),
            "environment_specific": {
                environment_id: self._build_environment_structure(environment_id)
            },
            "transfer_mappings": self._create_transfer_mapping_section(),
            "symbolic_decision_history": [],
            "metadata": {
                "version": "2.2.0 - Persistent Environment Knowledge",
                "agent_id": agent_id,
                "environments": [environment_id],
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "transfer_learning_enabled": True,
                "neural_symbolic_integration": True,
                "environment_knowledge_integration": True  # NEW
            }
        }

        # Load existing knowledge and canonical corpus
        self.load_knowledge()
        self._load_canonical_knowledge()
        self._load_default_template()

        print(f"🧩 Enhanced Agent Knowledge initialized with Environment Profile Integration")

    def _create_general_knowledge_section(self) -> Dict[str, Any]:
        return {
            "transferable_strategies": [],
            "meta_learning_principles": [],
            "cross_environment_patterns": [],
            "abstract_concepts": [],
            "neural_symbolic_correlations": []
        }

    def _create_transfer_mapping_section(self) -> Dict[str, Any]:
        return {
            "strategy_abstractions": {},
            "concept_translations": {},
            "success_patterns": [],
            "neural_pattern_mappings": {}
        }

    def _create_progression_block(self, unlock_defs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        return {
            "metrics": {
                "matches_played": 0,
                "wins": 0,
                "cumulative_reward": 0.0,
                "best_pattern_stability": 0.0,
                "best_symbolic_coherence": 0.0,
                "last_reward": 0.0
            },
            "unlocked": [],
            "unlock_history": [],
            "last_unlock_at": None,
            "unlock_definitions": unlock_defs or []
        }

    def _build_environment_structure(self, environment_id: str) -> Dict[str, Any]:
        return {
            "environment_profile": {
                "environment_id": environment_id,
                "display_name": self._get_environment_display_name(environment_id),
                "environment_type": "unknown",
                "understanding_level": "basic",
                "total_sessions": 0,
                "first_encountered": time.time(),
                "last_updated": time.time()
            },
            "objectives": {
                "primary": "",
                "secondary": [],
                "victory_conditions": [],
                "failure_conditions": []
            },
            "rules": {
                "core_mechanics": [],
                "constraints": [],
                "scoring": [],
                "special_conditions": []
            },
            "strategic_framework": {
                "core_skills_required": [],
                "winning_strategies": [],
                "success_patterns": [],
                "failure_patterns": [],
                "recommended_focus": []
            },
            "strategies": [],
            "lessons": [],
            "tactical_knowledge": [],
            "performance_patterns": [],
            "neural_insights": [],
            "experiment_logs": [],
            "knowledge_unlocks": [],
            "knowledge_progression": self._create_progression_block()
        }

    def _create_canonical_template(self) -> Dict[str, Any]:
        return {
            "general_knowledge": self._create_general_knowledge_section(),
            "environment_specific": {},
            "transfer_mappings": self._create_transfer_mapping_section(),
            "symbolic_decision_history": [],
            "experiment_logs": [],
            "metadata": {
                "version": "2.2.0 - Canonical Knowledge Corpus",
                "agent_id": self.agent_id,
                "environments": [],
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
        }

    def _load_canonical_knowledge(self):
        try:
            if self.canonical_knowledge_file and os.path.exists(self.canonical_knowledge_file):
                with open(self.canonical_knowledge_file, 'r') as f:
                    self.canonical_knowledge = json.load(f)
            else:
                # Ensure directory exists before first save
                if self.canonical_knowledge_file:
                    os.makedirs(os.path.dirname(self.canonical_knowledge_file), exist_ok=True)
                self.canonical_knowledge = self._create_canonical_template()
                self._save_canonical_knowledge()
        except Exception as e:
            print(f"⚠️ Could not load canonical knowledge corpus: {e}")
            self.canonical_knowledge = self._create_canonical_template()

        self._link_canonical_sections()
        self._sync_environment_to_canonical(initial=True)

    def _save_canonical_knowledge(self):
        try:
            if not self.canonical_knowledge_file:
                return
            os.makedirs(os.path.dirname(self.canonical_knowledge_file), exist_ok=True)
            self.canonical_knowledge['metadata']['last_updated'] = datetime.now().isoformat()
            with open(self.canonical_knowledge_file, 'w') as f:
                json.dump(self.canonical_knowledge, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save canonical knowledge corpus: {e}")

    def _link_canonical_sections(self):
        """Ensure general sections reference canonical corpus."""
        canonical_general = self.canonical_knowledge.setdefault('general_knowledge', self._create_general_knowledge_section())
        self.knowledge['general_knowledge'] = canonical_general

        canonical_transfer = self.canonical_knowledge.setdefault('transfer_mappings', self._create_transfer_mapping_section())
        self.knowledge['transfer_mappings'] = canonical_transfer

        canonical_decisions = self.canonical_knowledge.setdefault('symbolic_decision_history', [])
        self.knowledge['symbolic_decision_history'] = canonical_decisions

    def _load_default_template(self):
        if not self.default_knowledge_file:
            self.default_template = deepcopy(self.knowledge)
            return

        try:
            if os.path.exists(self.default_knowledge_file):
                with open(self.default_knowledge_file, 'r') as f:
                    self.default_template = json.load(f)
            else:
                self.default_template = deepcopy(self.knowledge)
                self._save_default_template()
        except Exception as e:
            print(f"⚠️ Could not load default environment knowledge: {e}")
            self.default_template = deepcopy(self.knowledge)

    def _save_default_template(self):
        if not self.default_knowledge_file or self.default_template is None:
            return
        try:
            os.makedirs(os.path.dirname(self.default_knowledge_file), exist_ok=True)
            self.default_template.setdefault('metadata', {})['last_updated'] = datetime.now().isoformat()
            with open(self.default_knowledge_file, 'w') as f:
                json.dump(self.default_template, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save default environment knowledge: {e}")

    def _get_default_env_entry(self) -> Optional[Dict[str, Any]]:
        if self.default_template is None:
            return None
        envs = self.default_template.setdefault('environment_specific', {})
        if self.environment_id not in envs:
            envs[self.environment_id] = deepcopy(self.knowledge['environment_specific'][self.environment_id])
        return envs[self.environment_id]

    def _sync_environment_to_canonical(self, initial: bool = False):
        if not self.canonical_knowledge_file:
            return

        env_data = self.knowledge['environment_specific'][self.environment_id]
        canonical_envs = self.canonical_knowledge.setdefault('environment_specific', {})
        canonical_envs[self.environment_id] = env_data

        metadata_envs = self.canonical_knowledge['metadata'].setdefault('environments', [])
        if self.environment_id not in metadata_envs:
            metadata_envs.append(self.environment_id)

        if not initial:
            self._save_canonical_knowledge()
        elif not os.path.exists(self.canonical_knowledge_file):
            self._save_canonical_knowledge()

    def _get_environment_display_name(self, environment_id: str) -> str:
        """Get human-readable name for environment"""
        display_names = {
            'arena_pong': 'Arena Pong',
            'pong': 'Classic Pong',
            'chess': 'Chess',
            'trading': 'Trading Simulator',
            'robotics': 'Robotics Environment'
        }
        return display_names.get(environment_id, environment_id.replace('_', ' ').title())

    def integrate_environment_profile(self, environment_id: str, env_context: Dict[str, Any]) -> bool:
        """
        🚀 CORE NEW METHOD: Permanently integrate environment understanding into knowledge base
        
        This ensures agents remember what each environment is about between sessions
        """
        try:
            # Ensure environment exists in knowledge
            if environment_id not in self.knowledge['environment_specific']:
                self.knowledge['environment_specific'][environment_id] = {
                    "environment_profile": {},
                    "objectives": {},
                    "rules": {},
                    "strategic_framework": {},
                    "strategies": [],
                    "lessons": [],
                    "tactical_knowledge": [],
                    "performance_patterns": [],
                    "neural_insights": []
                }

            env_knowledge = self.knowledge['environment_specific'][environment_id]

            # Update environment profile
            profile = env_knowledge.get('environment_profile', {})
            profile.update({
                'environment_id': environment_id,
                'display_name': self._get_environment_display_name(environment_id),
                'environment_type': env_context.get('environment_type', 'unknown'),
                'total_sessions': profile.get('total_sessions', 0) + 1,
                'last_updated': time.time()
            })

            # Set first encounter time if new
            if 'first_encountered' not in profile:
                profile['first_encountered'] = time.time()

            # Update understanding level based on experience
            sessions = profile['total_sessions']
            if sessions >= 15:
                profile['understanding_level'] = 'expert'
            elif sessions >= 10:
                profile['understanding_level'] = 'experienced'
            elif sessions >= 5:
                profile['understanding_level'] = 'intermediate'
            else:
                profile['understanding_level'] = 'basic'

            env_knowledge['environment_profile'] = profile

            # 🎯 Integrate objectives permanently
            if 'objective' in env_context:
                objectives = env_knowledge.get('objectives', {})
                
                # Handle both string and dict objective formats
                if isinstance(env_context['objective'], str):
                    objectives['primary'] = env_context['objective']
                elif isinstance(env_context['objective'], dict):
                    objectives.update({
                        'primary': env_context['objective'].get('primary', ''),
                        'secondary': env_context['objective'].get('secondary', [])
                    })
                
                # Add victory/failure conditions if provided
                if 'victory_conditions' in env_context:
                    objectives['victory_conditions'] = env_context['victory_conditions']
                if 'failure_conditions' in env_context:
                    objectives['failure_conditions'] = env_context['failure_conditions']
                    
                env_knowledge['objectives'] = objectives

            # 📋 Integrate rules permanently
            if 'rules' in env_context:
                rules = env_knowledge.get('rules', {})
                
                if isinstance(env_context['rules'], list):
                    rules['core_mechanics'] = env_context['rules']
                elif isinstance(env_context['rules'], dict):
                    rules.update(env_context['rules'])
                
                # Add additional rule categories if provided
                for rule_type in ['constraints', 'scoring', 'special_conditions']:
                    if rule_type in env_context:
                        rules[rule_type] = env_context[rule_type]
                        
                env_knowledge['rules'] = rules

            # 🧠 Integrate strategic framework
            if 'strategic_concepts' in env_context:
                strategic = env_context['strategic_concepts']
                framework = env_knowledge.get('strategic_framework', {})
                
                framework.update({
                    'core_skills_required': strategic.get('core_skills', []),
                    'winning_strategies': strategic.get('tactical_approaches', []),
                    'success_patterns': strategic.get('success_patterns', []),
                    'recommended_focus': strategic.get('recommended_focus', [])
                })
                
                env_knowledge['strategic_framework'] = framework

            # 📈 Capture unlock definitions for progression system
            if 'knowledge_unlocks' in env_context:
                env_knowledge['knowledge_unlocks'] = env_context.get('knowledge_unlocks', [])
                progression = self._ensure_progression_structure(env_knowledge)
                progression['unlock_definitions'] = env_knowledge['knowledge_unlocks']

            # 🔄 Add transferable skills to environment
            if 'transferable_skills' in env_context:
                if 'transferable_skills' not in env_knowledge:
                    env_knowledge['transferable_skills'] = []
                
                for skill in env_context['transferable_skills']:
                    if skill not in env_knowledge['transferable_skills']:
                        env_knowledge['transferable_skills'].append(skill)

            # 📚 Integrate learning recommendations
            if 'learning_recommendations' in env_context:
                learning_rec = env_context['learning_recommendations']
                framework = env_knowledge.get('strategic_framework', {})
                
                # Store learning focus areas
                focus_areas = []
                for focus_type, focus_items in learning_rec.items():
                    if isinstance(focus_items, list):
                        focus_areas.extend(focus_items)
                    
                framework['recommended_focus'] = focus_areas
                env_knowledge['strategic_framework'] = framework

            # Update environments list in metadata
            if environment_id not in self.knowledge['metadata']['environments']:
                self.knowledge['metadata']['environments'].append(environment_id)

            # Save the integrated knowledge
            self.save_knowledge()

            print(f"🧩 Environment profile integrated for {environment_id}")
            print(f"   📊 Understanding level: {profile['understanding_level']}")
            print(f"   🎯 Total sessions: {profile['total_sessions']}")
            print(f"   ✅ Objectives stored: {bool(env_knowledge.get('objectives', {}).get('primary'))}")
            print(f"   📋 Rules stored: {len(env_knowledge.get('rules', {}).get('core_mechanics', []))}")
            print(f"   🧠 Strategic framework: {len(env_knowledge.get('strategic_framework', {}).get('core_skills_required', []))} skills")

            return True

        except Exception as e:
            print(f"❌ Error integrating environment profile: {e}")
            return False

    def load_environment_understanding(self, environment_id: str) -> Dict[str, Any]:
        """
        🧠 Load persistent environment understanding from knowledge base
        
        Returns comprehensive understanding that agent has built up over time
        """
        try:
            env_knowledge = self.knowledge['environment_specific'].get(environment_id, {})
            
            if not env_knowledge:
                print(f"⚠️ No knowledge found for {environment_id}")
                return {}

            profile = env_knowledge.get('environment_profile', {})
            
            # Build comprehensive understanding report
            understanding = {
                'environment_id': environment_id,
                'display_name': profile.get('display_name', environment_id),
                'understanding_level': profile.get('understanding_level', 'basic'),
                'total_experience': profile.get('total_sessions', 0),
                'environment_type': profile.get('environment_type', 'unknown'),
                
                # Knowledge flags
                'knows_objectives': bool(env_knowledge.get('objectives', {}).get('primary')),
                'knows_rules': bool(env_knowledge.get('rules', {}).get('core_mechanics')),
                'has_strategic_framework': bool(env_knowledge.get('strategic_framework', {}).get('core_skills_required')),
                'has_experience': profile.get('total_sessions', 0) > 0,
                
                # Detailed knowledge
                'objectives': env_knowledge.get('objectives', {}),
                'rules': env_knowledge.get('rules', {}),
                'strategic_framework': env_knowledge.get('strategic_framework', {}),
                'transferable_skills': env_knowledge.get('transferable_skills', []),
                
                # Agent-learned knowledge
                'strategies': env_knowledge.get('strategies', []),
                'lessons': env_knowledge.get('lessons', []),
                'neural_insights': env_knowledge.get('neural_insights', []),
                'performance_patterns': env_knowledge.get('performance_patterns', []),
                
                # Timestamps
                'first_encountered': profile.get('first_encountered'),
                'last_updated': profile.get('last_updated')
            }

            # Generate experience summary
            experience_level = understanding['understanding_level']
            experience_msg = {
                'basic': f"🆕 Learning about {understanding['display_name']}",
                'intermediate': f"🎓 Developing understanding of {understanding['display_name']}",
                'experienced': f"🧠 Experienced with {understanding['display_name']}",
                'expert': f"🏆 Expert level understanding of {understanding['display_name']}"
            }

            print(f"🧠 Loaded environment understanding for {environment_id}")
            print(f"   {experience_msg.get(experience_level, 'Unknown level')}")
            print(f"   📊 Sessions: {understanding['total_experience']}")
            print(f"   🎯 Knows objectives: {understanding['knows_objectives']}")
            print(f"   📋 Knows rules: {understanding['knows_rules']}")
            print(f"   🧠 Has strategy framework: {understanding['has_strategic_framework']}")
            
            return understanding

        except Exception as e:
            print(f"❌ Error loading environment understanding: {e}")
            return {}

    def get_environment_knowledge_summary(self, environment_id: str) -> str:
        """Generate human-readable summary of what agent knows about environment"""
        understanding = self.load_environment_understanding(environment_id)
        
        if not understanding:
            return f"No knowledge about {environment_id}"

        # Build summary
        summary_parts = []
        
        # Basic info
        display_name = understanding.get('display_name', environment_id)
        level = understanding.get('understanding_level', 'basic')
        sessions = understanding.get('total_experience', 0)
        
        summary_parts.append(f"Environment: {display_name} (Level: {level}, Sessions: {sessions})")
        
        # Objectives
        objectives = understanding.get('objectives', {})
        if objectives.get('primary'):
            summary_parts.append(f"Objective: {objectives['primary']}")
        
        # Rules
        rules = understanding.get('rules', {})
        core_mechanics = rules.get('core_mechanics', [])
        if core_mechanics:
            rule_names = [str(rule) for rule in core_mechanics[:3]]
            summary_parts.append(f"Key Rules: {', '.join(rule_names)}")
        
        # Strategic framework
        framework = understanding.get('strategic_framework', {})
        skills = framework.get('core_skills_required', [])
        if skills:
            skill_names = [str(skill) for skill in skills[:3]]
            summary_parts.append(f"Core Skills: {', '.join(skill_names)}")
        
        # Transferable skills
        transferable = understanding.get('transferable_skills', [])
        if transferable:
            skill_names = [str(skill) for skill in transferable[:3]]
            summary_parts.append(f"Transferable Skills: {', '.join(skill_names)}")
        
        return " | ".join(summary_parts)

    def migrate_legacy_knowledge(self) -> bool:
        """Migrate legacy knowledge format to include environment profiles"""
        try:
            migrated = False
            
            for env_id, env_data in self.knowledge['environment_specific'].items():
                # Check if environment profile exists
                if 'environment_profile' not in env_data:
                    # Create basic profile from existing data
                    env_data['environment_profile'] = {
                        'environment_id': env_id,
                        'display_name': self._get_environment_display_name(env_id),
                        'environment_type': 'migrated',
                        'understanding_level': 'intermediate' if len(env_data.get('strategies', [])) > 5 else 'basic',
                        'total_sessions': len(env_data.get('performance_patterns', [])),
                        'first_encountered': time.time() - (86400 * 30),  # Assume 30 days ago
                        'last_updated': time.time()
                    }
                    migrated = True
                
                # Add missing knowledge sections
                for section in ['objectives', 'rules', 'strategic_framework']:
                    if section not in env_data:
                        env_data[section] = {}
                        migrated = True
            
            # Update metadata if we migrated anything
            if migrated:
                self.knowledge['metadata']['environment_knowledge_integration'] = True
                self.knowledge['metadata']['migrated_at'] = datetime.now().isoformat()
                self.save_knowledge()
                print("🔄 Legacy knowledge migrated to include environment profiles")
            
            return migrated
            
        except Exception as e:
            print(f"❌ Error migrating legacy knowledge: {e}")
            return False

    def _create_new_environment_entry(self, environment_id: str):
        """Create new environment entry in knowledge structure"""
        if environment_id not in self.knowledge['environment_specific']:
            self.knowledge['environment_specific'][environment_id] = self._build_environment_structure(environment_id)
            print(f"🆕 Created new environment entry for {environment_id}")
        else:
            env_knowledge = self.knowledge['environment_specific'][environment_id]
            # Backfill new fields if missing
            env_knowledge.setdefault('experiment_logs', [])
            env_knowledge.setdefault('knowledge_unlocks', [])
            env_knowledge.setdefault(
                'knowledge_progression',
                self._create_progression_block(env_knowledge.get('knowledge_unlocks', []))
            )

    def _ensure_progression_structure(self, env_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        progression = env_knowledge.get('knowledge_progression')
        if not progression:
            progression = self._create_progression_block(env_knowledge.get('knowledge_unlocks', []))
            env_knowledge['knowledge_progression'] = progression

        metrics = progression.setdefault('metrics', {
            "matches_played": 0,
            "wins": 0,
            "cumulative_reward": 0.0,
            "best_pattern_stability": 0.0,
            "best_symbolic_coherence": 0.0,
            "last_reward": 0.0
        })

        # Ensure all metric keys exist
        for key in ["matches_played", "wins", "cumulative_reward", "best_pattern_stability",
                    "best_symbolic_coherence", "last_reward"]:
            metrics.setdefault(key, 0 if 'reward' not in key else 0.0)

        progression.setdefault('unlocked', [])
        progression.setdefault('unlock_history', [])
        progression.setdefault('last_unlock_at', None)
        progression.setdefault('unlock_definitions', env_knowledge.get('knowledge_unlocks', []))

        # Ensure default template has a matching environment entry
        self._get_default_env_entry()

        return progression

    def update_environment_progression(self, outcome: Dict[str, Any]):
        """Update knowledge progression metrics and unlock new knowledge when thresholds are met."""
        try:
            env_knowledge = self.knowledge['environment_specific'][self.environment_id]
            progression = self._ensure_progression_structure(env_knowledge)
            metrics = progression['metrics']

            metrics['matches_played'] += 1
            if outcome.get('win'):
                metrics['wins'] += 1

            reward = outcome.get('reward', 0.0) or 0.0
            metrics['cumulative_reward'] += reward
            metrics['last_reward'] = reward

            pattern_stability = outcome.get('pattern_stability', 0.0) or 0.0
            metrics['best_pattern_stability'] = max(metrics.get('best_pattern_stability', 0.0), pattern_stability)

            symbolic_coherence = outcome.get('symbolic_coherence', 0.0) or 0.0
            metrics['best_symbolic_coherence'] = max(metrics.get('best_symbolic_coherence', 0.0), symbolic_coherence)

            progression['last_update'] = datetime.now().isoformat()
            self._apply_unlocks(env_knowledge, progression)
            self._sync_environment_to_canonical()

        except Exception as e:
            print(f"⚠️ Unable to update knowledge progression: {e}")

    def record_experiment_outcome(self, experiment: Dict[str, Any]):
        """Record symbolic/neural experiment outcomes for dataset-ready tracing."""
        try:
            env_knowledge = self.knowledge['environment_specific'][self.environment_id]
            experiment_entry = {
                'timestamp': experiment.get('timestamp', time.time()),
                'environment': experiment.get('environment', self.environment_id),
                'strategy': experiment.get('strategy'),
                'decision_type': experiment.get('decision_type'),
                'reward': experiment.get('reward', 0.0),
                'success': bool(experiment.get('success')),
                'expert_choice': experiment.get('expert_choice'),
                'notes': experiment.get('notes')
            }

            env_logs = env_knowledge.setdefault('experiment_logs', [])
            env_logs.append(experiment_entry)
            if len(env_logs) > 200:
                env_knowledge['experiment_logs'] = env_logs[-200:]

            canonical_logs = self.canonical_knowledge.setdefault('experiment_logs', [])
            canonical_logs.append(experiment_entry)
            if len(canonical_logs) > 2000:
                self.canonical_knowledge['experiment_logs'] = canonical_logs[-2000:]

            self._sync_environment_to_canonical()
        except Exception as e:
            print(f"⚠️ Unable to log experiment outcome: {e}")

    def _apply_unlocks(self, env_knowledge: Dict[str, Any], progression: Dict[str, Any]):
        unlocks = progression.get('unlock_definitions', [])
        unlocked_ids = progression.setdefault('unlocked', [])

        for unlock in unlocks:
            unlock_id = unlock.get('id')
            if not unlock_id or unlock_id in unlocked_ids:
                continue

            requirements = unlock.get('requirements', {})
            if not self._requirements_met(requirements, progression['metrics']):
                continue

            self._merge_unlock_grants(env_knowledge, unlock.get('grants', {}))
            unlocked_ids.append(unlock_id)
            history = progression.setdefault('unlock_history', [])
            history.append({
                'id': unlock_id,
                'timestamp': datetime.now().isoformat(),
                'metrics': deepcopy(progression['metrics'])
            })
            progression['last_unlock_at'] = datetime.now().isoformat()
            print(f"🔓 Unlocked new knowledge tier '{unlock_id}' for {self.environment_id}")

    def _requirements_met(self, requirements: Dict[str, Any], metrics: Dict[str, Any]) -> bool:
        if not requirements:
            return True

        if 'wins' in requirements and metrics.get('wins', 0) < requirements['wins']:
            return False
        if 'matches_played' in requirements and metrics.get('matches_played', 0) < requirements['matches_played']:
            return False
        if 'cumulative_reward' in requirements and metrics.get('cumulative_reward', 0.0) < requirements['cumulative_reward']:
            return False
        if 'pattern_stability' in requirements and metrics.get('best_pattern_stability', 0.0) < requirements['pattern_stability']:
            return False
        if 'symbolic_coherence' in requirements and metrics.get('best_symbolic_coherence', 0.0) < requirements['symbolic_coherence']:
            return False
        return True

    def _merge_unlock_grants(self, env_knowledge: Dict[str, Any], grants: Dict[str, Any]):
        if not grants:
            return

        self._apply_grants_to_env(env_knowledge, grants)
        self._merge_grants_into_default(grants)

    def _apply_grants_to_env(self, env_knowledge: Dict[str, Any], grants: Dict[str, Any]):
        if 'rules' in grants:
            env_rules = env_knowledge.setdefault('rules', {})
            for key, values in grants['rules'].items():
                env_rules.setdefault(key, [])
                for value in values:
                    self._append_unique(env_rules[key], value)

        if 'objectives' in grants:
            env_objectives = env_knowledge.setdefault('objectives', {})
            for key, values in grants['objectives'].items():
                if isinstance(values, list):
                    env_objectives.setdefault(key, [])
                    for value in values:
                        self._append_unique(env_objectives[key], value)
                else:
                    env_objectives[key] = values

        if 'strategic_framework' in grants:
            env_framework = env_knowledge.setdefault('strategic_framework', {})
            for key, values in grants['strategic_framework'].items():
                env_framework.setdefault(key, [])
                for value in values:
                    self._append_unique(env_framework[key], value)

        if 'learning_recommendations' in grants:
            env_learning = env_knowledge.setdefault('learning_recommendations', {})
            for key, values in grants['learning_recommendations'].items():
                env_learning.setdefault(key, [])
                for value in values:
                    self._append_unique(env_learning[key], value)

    def _merge_grants_into_default(self, grants: Dict[str, Any]):
        if not self.default_template or not self.default_knowledge_file or not grants:
            return

        default_env = self._get_default_env_entry()
        if not default_env:
            return

        self._apply_grants_to_env(default_env, grants)
        self._save_default_template()

    @staticmethod
    def _append_unique(collection: List[Any], value: Any):
        if value is None:
            return
        if value not in collection:
            collection.append(value)

    def integrate_neural_insights(self, neural_insights: Dict[str, Any]):
        """Integrate insights from the neural brain into symbolic knowledge"""
        try:
            env_knowledge = self.knowledge['environment_specific'][self.environment_id]

            # Store neural insights
            if 'neural_insights' not in env_knowledge:
                env_knowledge['neural_insights'] = []

            neural_insight_record = {
                'timestamp': time.time(),
                'neural_skill_development': neural_insights.get('neural_skill_development', {}),
                'learning_trajectory': neural_insights.get('learning_trajectory', {}),
                'transfer_readiness': neural_insights.get('transfer_readiness', 0.0),
                'action_reward_correlations': neural_insights.get('action_reward_correlations', {})
            }

            env_knowledge['neural_insights'].append(neural_insight_record)

            # Keep recent insights
            if len(env_knowledge['neural_insights']) > 20:
                env_knowledge['neural_insights'] = env_knowledge['neural_insights'][-20:]

            # Extract transferable skills from neural development
            neural_skills = neural_insights.get('neural_skill_development', {})
            for skill, skill_data in neural_skills.items():
                if skill_data.get('best_confidence', 0) > 0.6:  # High confidence threshold
                    self._add_neural_derived_strategy(skill, skill_data)

            # Update neural-symbolic correlations
            self._update_neural_symbolic_correlations(neural_insights)

            print(f"🔗 Neural insights integrated: {len(neural_skills)} skills analyzed")

        except Exception as e:
            print(f"❌ Error integrating neural insights: {e}")

    def _add_neural_derived_strategy(self, skill: str, skill_data: Dict[str, Any]):
        """Add strategy derived from neural skill development"""
        strategy = f"Neural-derived {skill} strategy (confidence: {skill_data.get('best_confidence', 0):.2f})"

        env_knowledge = self.knowledge['environment_specific'][self.environment_id]
        if strategy not in env_knowledge['strategies']:
            env_knowledge['strategies'].append(strategy)

        # Also add to transferable strategies if confidence is high
        if skill_data.get('best_confidence', 0) > 0.8:
            transferable_strategy = {
                'strategy': f"Universal {skill} pattern",
                'source_environment': self.environment_id,
                'confidence': skill_data.get('best_confidence', 0),
                'neural_derived': True,
                'created_at': time.time(),
                'usage_count': 0,
                'success_rate': 0.0
            }

            general_strategies = self.knowledge['general_knowledge']['transferable_strategies']
            if not any(s['strategy'] == transferable_strategy['strategy'] for s in general_strategies):
                general_strategies.append(transferable_strategy)

    def _update_neural_symbolic_correlations(self, neural_insights: Dict[str, Any]):
        """Update correlations between neural patterns and symbolic concepts"""
        correlations = self.knowledge['general_knowledge']['neural_symbolic_correlations']

        # Map neural skills to symbolic concepts
        neural_skills = neural_insights.get('neural_skill_development', {})
        for skill, skill_data in neural_skills.items():
            correlation_record = {
                'neural_skill': skill,
                'symbolic_concept': self._map_neural_skill_to_concept(skill),
                'confidence': skill_data.get('best_confidence', 0),
                'environment': self.environment_id,
                'timestamp': time.time()
            }

            correlations.append(correlation_record)

        # Keep recent correlations
        if len(correlations) > 50:
            self.knowledge['general_knowledge']['neural_symbolic_correlations'] = correlations[-50:]

    def _map_neural_skill_to_concept(self, neural_skill: str) -> str:
        """Map neural skill to symbolic concept"""
        skill_to_concept = {
            'trajectory_prediction': 'dynamic_object_tracking',
            'timing_optimization': 'temporal_coordination',
            'strategic_positioning': 'spatial_optimization',
            'pattern_recognition': 'behavioral_modeling',
            'error_recovery': 'adaptive_resilience',
            'adaptive_strategy': 'contextual_adaptation'
        }
        return skill_to_concept.get(neural_skill, 'general_intelligence')

    def make_enhanced_symbolic_decision(self, neural_state: np.ndarray, neural_q_values: np.ndarray,
                                        neural_insights: Dict[str, Any], exploration_rate: float) -> Tuple[
        int, str, Dict]:
        """Make enhanced symbolic decision using neural-symbolic integration"""
        try:
            # Set environment context for symbolic decision maker
            env_context = self._get_current_environment_context()
            self.symbolic_decision_maker.set_environment_context(self.environment_id, env_context)

            # Get recent action/reward history from neural insights
            action_history = self._extract_action_history(neural_insights)
            reward_history = self._extract_reward_history(neural_insights)

            # Make enhanced decision with full neural-symbolic integration
            action, reasoning, decision_info = self.symbolic_decision_maker.make_enhanced_decision(
                neural_state, neural_q_values, action_history, reward_history, exploration_rate
            )

            # Record symbolic decision
            decision_record = {
                'timestamp': time.time(),
                'action': action,
                'reasoning': reasoning,
                'decision_info': decision_info,
                'neural_insights_used': bool(neural_insights),
                'environment': self.environment_id
            }

            if 'symbolic_decision_history' not in self.knowledge:
                self.knowledge['symbolic_decision_history'] = []

            self.knowledge['symbolic_decision_history'].append(decision_record)

            # Keep decision history manageable
            if len(self.knowledge['symbolic_decision_history']) > 100:
                self.knowledge['symbolic_decision_history'] = self.knowledge['symbolic_decision_history'][-50:]

            return action, reasoning, decision_info

        except Exception as e:
            print(f"❌ Error in enhanced symbolic decision: {e}")
            # Fallback to neural decision
            return int(np.argmax(neural_q_values)), "Fallback neural decision", {}

    def _get_current_environment_context(self) -> Dict[str, Any]:
        """Get current environment context for symbolic decision making"""
        env_knowledge = self.knowledge['environment_specific'][self.environment_id]

        return {
            'environment_id': self.environment_id,
            'strategies': env_knowledge.get('strategies', []),
            'lessons': env_knowledge.get('lessons', []),
            'transferable_skills': self._get_available_transferable_skills(),
            'neural_insights': env_knowledge.get('neural_insights', [])[-3:],  # Recent insights
            'performance_patterns': env_knowledge.get('performance_patterns', [])
        }

    def _get_available_transferable_skills(self) -> List[str]:
        """Get transferable skills available for this environment"""
        transferable_skills = []

        # From general knowledge
        for strategy in self.knowledge['general_knowledge']['transferable_strategies']:
            if strategy.get('confidence', 0) > 0.5:
                transferable_skills.append(strategy['strategy'])

        # From neural-symbolic correlations
        for correlation in self.knowledge['general_knowledge']['neural_symbolic_correlations']:
            if correlation.get('confidence', 0) > 0.6:
                transferable_skills.append(correlation['symbolic_concept'])

        return list(set(transferable_skills))  # Remove duplicates

    def _extract_action_history(self, neural_insights: Dict[str, Any]) -> List[Dict]:
        """Extract action history from neural insights"""
        recent_patterns = neural_insights.get('recent_neural_patterns', [])
        return [{'action': p.get('action', 0), 'timestamp': p.get('timestamp', time.time())}
                for p in recent_patterns]

    def _extract_reward_history(self, neural_insights: Dict[str, Any]) -> List[float]:
        """Extract reward history from neural insights"""
        recent_patterns = neural_insights.get('recent_neural_patterns', [])
        return [p.get('reward', 0.0) for p in recent_patterns]

    def update_symbolic_decision_outcome(self, decision_info: Dict[str, Any], reward: float):
        """Update symbolic decision outcome for learning"""
        if hasattr(self.symbolic_decision_maker, 'update_neural_symbolic_correlation'):
            self.symbolic_decision_maker.update_neural_symbolic_correlation(decision_info, reward)

    def record_game_result(self, app_name: str, win: bool, reward: float, outcome_summary: str):
        """Record game result in knowledge base"""
        try:
            if app_name not in self.knowledge['environment_specific']:
                self.knowledge['environment_specific'][app_name] = {
                    'strategies': [],
                    'lessons': [],
                    'tactical_knowledge': [],
                    'performance_patterns': [],
                    'transferable_strategies': []
                }
            
            # Record performance pattern
            performance_record = {
                'timestamp': time.time(),
                'win': win,
                'reward': reward,
                'outcome_summary': outcome_summary,
                'transferable_insights': []
            }
            
            self.knowledge['environment_specific'][app_name]['performance_patterns'].append(performance_record)
            
            # Keep recent patterns only
            patterns = self.knowledge['environment_specific'][app_name]['performance_patterns']
            if len(patterns) > 100:
                self.knowledge['environment_specific'][app_name]['performance_patterns'] = patterns[-100:]
                
            # Save updated knowledge
            self.save_knowledge()
            
        except Exception as e:
            print(f"❌ Error recording game result: {e}")

    def load_knowledge(self):
        """Enhanced load with automatic migration"""
        try:
            if os.path.exists(self.knowledge_file):
                with open(self.knowledge_file, 'r') as f:
                    loaded_knowledge = json.load(f)

                # Check if migration needed
                if 'environment_knowledge_integration' not in loaded_knowledge.get('metadata', {}):
                    print("🔄 Migrating knowledge to include environment profiles...")
                    self.knowledge = loaded_knowledge
                    self.migrate_legacy_knowledge()
                else:
                    self.knowledge = loaded_knowledge

                # Ensure current environment exists
                if self.environment_id not in self.knowledge['environment_specific']:
                    self._create_new_environment_entry(self.environment_id)

                # Update metadata
                envs = self.knowledge['metadata'].get('environments', [])
                if self.environment_id not in envs:
                    envs.append(self.environment_id)
                    self.knowledge['metadata']['environments'] = envs

                print(f"🧩 Enhanced knowledge loaded with environment profile integration")
                env_entry = self.knowledge['environment_specific'][self.environment_id]
                self._ensure_progression_structure(env_entry)
                return True
            else:
                print(f"🆕 No existing knowledge found, starting fresh with environment profiles")
                self._create_new_environment_entry(self.environment_id)
                env_entry = self.knowledge['environment_specific'][self.environment_id]
                self._ensure_progression_structure(env_entry)
                return False

        except Exception as e:
            print(f"❌ Error loading enhanced knowledge: {e}")
            self._create_new_environment_entry(self.environment_id)
            env_entry = self.knowledge['environment_specific'][self.environment_id]
            self._ensure_progression_structure(env_entry)
            return False


    def save_knowledge(self):
        """Save enhanced knowledge with neural-symbolic integration"""
        try:
            os.makedirs(os.path.dirname(self.knowledge_file), exist_ok=True)
            self.knowledge['metadata']['last_updated'] = datetime.now().isoformat()

            with open(self.knowledge_file, 'w') as f:
                json.dump(self.knowledge, f, indent=2)

            neural_correlations = len(self.knowledge['general_knowledge']['neural_symbolic_correlations'])
            decisions_tracked = len(self.knowledge.get('symbolic_decision_history', []))

            print(
                f"🧩 Enhanced knowledge saved: {neural_correlations} neural correlations, {decisions_tracked} decisions tracked")
            self._sync_environment_to_canonical()
            return True

        except Exception as e:
            print(f"❌ Error saving enhanced knowledge: {e}")
            return False

    def get_neural_symbolic_integration_report(self) -> Dict[str, Any]:
        """Get comprehensive report on neural-symbolic integration"""
        try:
            env_knowledge = self.knowledge['environment_specific'][self.environment_id]
            general_knowledge = self.knowledge['general_knowledge']

            return {
                'environment_id': self.environment_id,
                'neural_insights_collected': len(env_knowledge.get('neural_insights', [])),
                'neural_symbolic_correlations': len(general_knowledge.get('neural_symbolic_correlations', [])),
                'symbolic_decisions_made': len(self.knowledge.get('symbolic_decision_history', [])),
                'transferable_strategies': len(general_knowledge.get('transferable_strategies', [])),
                'neural_derived_strategies': len([s for s in env_knowledge.get('strategies', [])
                                                  if 'Neural-derived' in s]),
                'integration_quality': self._calculate_integration_quality(),
                'transfer_readiness': self._calculate_symbolic_transfer_readiness()
            }

        except Exception as e:
            print(f"❌ Error generating integration report: {e}")
            return {}

    def _calculate_integration_quality(self) -> float:
        """Calculate quality of neural-symbolic integration"""
        try:
            # Quality based on correlation strength and decision success
            correlations = self.knowledge['general_knowledge'].get('neural_symbolic_correlations', [])
            if not correlations:
                return 0.0

            avg_correlation_confidence = np.mean([c.get('confidence', 0) for c in correlations])

            decisions = self.knowledge.get('symbolic_decision_history', [])
            if decisions:
                recent_decisions = decisions[-20:]
                neural_informed_decisions = [d for d in recent_decisions if d.get('neural_insights_used', False)]
                neural_integration_rate = len(neural_informed_decisions) / len(recent_decisions)
            else:
                neural_integration_rate = 0.0

            return (avg_correlation_confidence * 0.6 + neural_integration_rate * 0.4)

        except Exception as e:
            print(f"❌ Error calculating integration quality: {e}")
            return 0.0

    def _calculate_symbolic_transfer_readiness(self) -> float:
        """Calculate readiness for symbolic knowledge transfer"""
        try:
            transferable_strategies = self.knowledge['general_knowledge'].get('transferable_strategies', [])
            if not transferable_strategies:
                return 0.0

            # Readiness based on strategy confidence and neural support
            high_confidence_strategies = [s for s in transferable_strategies if s.get('confidence', 0) > 0.7]
            neural_supported_strategies = [s for s in transferable_strategies if s.get('neural_derived', False)]

            strategy_quality = len(high_confidence_strategies) / len(transferable_strategies)
            neural_support = len(neural_supported_strategies) / len(transferable_strategies)

            return (strategy_quality * 0.7 + neural_support * 0.3)

        except Exception as e:
            print(f"❌ Error calculating symbolic transfer readiness: {e}")
            return 0.0


class EnhancedDualBrainAgent:
    """Enhanced Dual Brain System with Neural-Symbolic Integration and Transfer Learning"""

    def __init__(self, agent_id: str = None, environment_id: str = "unknown",
                 brain_file: str = None, knowledge_file: str = None,
                 canonical_knowledge_file: Optional[str] = None,
                 default_knowledge_file: Optional[str] = None):

        # Handle backwards compatibility
        if agent_id is None:
            agent_id = f"agent_{int(time.time())}"
        if environment_id == "unknown":
            environment_id = "pong"  # Default environment

        # Set up file paths for multi-environment structure
        if brain_file is None:
            brain_file = f"saas_agents/{agent_id}/environments/{environment_id}/brain.json"
        if knowledge_file is None:
            knowledge_file = f"saas_agents/{agent_id}/environments/{environment_id}/{environment_id}_knowledge.json"

        if canonical_knowledge_file is None:
            env_dir = os.path.dirname(knowledge_file)
            agent_root = os.path.dirname(os.path.dirname(env_dir)) if env_dir else ""
            if agent_root:
                canonical_knowledge_file = os.path.join(agent_root, "core", "knowledge.json")
            else:
                canonical_knowledge_file = knowledge_file

        if default_knowledge_file is None:
            env_dir = os.path.dirname(knowledge_file)
            default_knowledge_file = os.path.join(
                env_dir, f"default_{environment_id}_knowledge.json") if env_dir else knowledge_file

        # Initialize enhanced dual brain system
        self.agent_id = agent_id
        self.environment_id = environment_id
        self.brain = EnhancedAgentBrain(agent_id, environment_id, brain_file)
        self.knowledge = EnhancedAgentKnowledge(
            agent_id,
            environment_id,
            knowledge_file,
            canonical_knowledge_file=canonical_knowledge_file,
            default_knowledge_file=default_knowledge_file
        )

        # Current session state
        self.current_app = None
        self.current_context = None
        self.session_transfer_events = []
        self.neural_symbolic_integration_active = True

        # Integration tracking
        self.integration_history = []
        self.decision_correlation_history = []

        print("🧠🧩 Enhanced Dual Brain Agent with Neural-Symbolic Integration initialized!")
        print(f"   🆔 Agent ID: {agent_id}")
        print(f"   🌍 Environment: {environment_id}")
        print(f"   🧠 Enhanced Neural Brain: {self.brain.total_training_steps} total steps")
        print(f"   🧩 Enhanced Symbolic Knowledge: Neural-symbolic integration active")
        print(f"   🔄 Transfer Learning: Multi-environment capable")

    def _load_app_context(self, app_name: str, category: str = "games", env_context: Optional[Dict] = None) -> Dict:
        """Load application context for the session"""
        try:
            # Use environment context if provided, otherwise create basic context
            if env_context:
                return env_context
            else:
                # Create basic context for the environment
                return {
                    'environment_id': app_name,
                    'strategies': [],
                    'transferable_skills': [],
                    'lessons': [],
                    'category': category
                }
        except Exception as e:
            print(f"❌ Error loading app context: {e}")
            return {}

    def start_enhanced_session(self, app_name: str, category: str = "games", 
                              env_context: Optional[Dict] = None, 
                              existing_knowledge: Optional[Dict] = None):
        """
        🚀 Enhanced session start with persistent environment knowledge
        """
        self.current_app = app_name
        self.session_transfer_events = []

        # 🧠 Use existing knowledge if available, otherwise load context
        if existing_knowledge:
            print(f"🧠 Using existing environment knowledge for {app_name}")
            # Convert existing knowledge back to context format for compatibility
            self.current_context = self._convert_knowledge_to_context(existing_knowledge, env_context)
        else:
            # Load context with enhanced symbolic processing
            self.current_context = self._load_app_context(app_name, category, env_context)

        # Initialize neural-symbolic integration for this session
        if self.neural_symbolic_integration_active:
            self._initialize_session_integration()

        # Show enhanced session start info
        print(f"🚀 Enhanced session started: {app_name}")
        if existing_knowledge:
            level = existing_knowledge.get('understanding_level', 'basic')
            sessions = existing_knowledge.get('total_experience', 0)
            print(f"   🧠 Agent understanding level: {level} ({sessions} sessions)")
            
            if existing_knowledge.get('knows_objectives'):
                objective = existing_knowledge.get('objectives', {}).get('primary', '')
                print(f"   🎯 Known objective: {objective}")
        
        if self.current_context:
            env_strategies = len(self.current_context.get('strategies', []))
            transferable_strategies = len(self.current_context.get('transferable_strategies', []))
            neural_insights = len(
                self.knowledge.knowledge['environment_specific'][self.environment_id].get('neural_insights', []))

            print(f"   📚 Environment strategies: {env_strategies}")
            print(f"   🔄 Transferable strategies: {transferable_strategies}")
            print(f"   🧠 Neural insights: {neural_insights}")
            print(f"   🔗 Neural-symbolic integration: Active")

        return self.current_context

    def _initialize_session_integration(self):
        """Initialize neural-symbolic integration for session"""
        try:
            # Get neural insights from brain
            neural_insights = self.brain.get_neural_insights_for_symbolic_brain()

            # Integrate insights into symbolic knowledge
            self.knowledge.integrate_neural_insights(neural_insights)

            # Record integration event
            integration_event = {
                'timestamp': time.time(),
                'session_app': self.current_app,
                'neural_skills_detected': len(neural_insights.get('neural_skill_development', {})),
                'transfer_readiness': neural_insights.get('transfer_readiness', 0.0),
                'integration_quality': self.knowledge._calculate_integration_quality()
            }

            self.integration_history.append(integration_event)

            # Keep integration history manageable
            if len(self.integration_history) > 50:
                self.integration_history = self.integration_history[-25:]

            print(f"🔗 Neural-symbolic integration initialized for session")

        except Exception as e:
            print(f"❌ Error initializing session integration: {e}")

    def _convert_knowledge_to_context(self, existing_knowledge: Dict[str, Any], 
                                     new_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        🔄 Convert stored knowledge back to context format for session compatibility
        """
        # Start with stored knowledge
        context = {
            'environment_id': existing_knowledge.get('environment_id', self.environment_id),
            'strategies': existing_knowledge.get('strategies', []),
            'transferable_skills': existing_knowledge.get('transferable_skills', []),
            'lessons': existing_knowledge.get('lessons', [])
        }
        
        # Add objectives if known
        objectives = existing_knowledge.get('objectives', {})
        if objectives.get('primary'):
            context['objective'] = {'primary': objectives['primary']}
            if objectives.get('secondary'):
                context['objective']['secondary'] = objectives['secondary']
        
        # Add rules if known  
        rules = existing_knowledge.get('rules', {})
        if rules.get('core_mechanics'):
            context['rules'] = rules['core_mechanics']
        
        # Add strategic framework if available
        framework = existing_knowledge.get('strategic_framework', {})
        if framework:
            context['strategic_concepts'] = {
                'core_skills': framework.get('core_skills_required', []),
                'tactical_approaches': framework.get('winning_strategies', []),
                'success_patterns': framework.get('success_patterns', [])
            }
        
        # Merge with any new context provided
        if new_context:
            # New context takes precedence for updates
            for key, value in new_context.items():
                if key not in context or not context[key]:
                    context[key] = value
        
        return context

    def make_enhanced_decision(self, neural_state: np.ndarray, neural_q_values: np.ndarray,
                               exploration_rate: float) -> Tuple[int, str, Dict]:
        """Make enhanced decision using full neural-symbolic integration"""
        try:
            # Get current neural insights
            neural_insights = self.brain.get_neural_insights_for_symbolic_brain()

            # Make symbolic decision with neural integration
            action, reasoning, decision_info = self.knowledge.make_enhanced_symbolic_decision(
                neural_state, neural_q_values, neural_insights, exploration_rate
            )

            # Track decision correlation
            correlation_record = {
                'timestamp': time.time(),
                'neural_action': int(np.argmax(neural_q_values)),
                'symbolic_action': action,
                'decision_type': decision_info.get('decision_type', 'unknown'),
                'neural_insights_available': bool(neural_insights.get('neural_skill_development')),
                'reasoning': reasoning
            }

            self.decision_correlation_history.append(correlation_record)

            # Keep correlation history manageable
            if len(self.decision_correlation_history) > 100:
                self.decision_correlation_history = self.decision_correlation_history[-50:]

            return action, reasoning, decision_info

        except Exception as e:
            print(f"❌ Error in enhanced decision making: {e}")
            # Fallback to neural decision
            return int(np.argmax(neural_q_values)), "Fallback neural decision", {'decision_type': 'neural_fallback'}

    def record_enhanced_learning_step(self, reward: float, loss: float, exploration_rate: float,
                                      neural_state: np.ndarray, action: int, decision_info: Optional[Dict] = None):
        """Record enhanced learning step with neural-symbolic integration"""

        # Record in neural brain with enhanced tracking
        self.brain.record_enhanced_learning_step(reward, loss, exploration_rate, neural_state, action)

        # Update symbolic knowledge based on decision outcome
        if decision_info:
            self.knowledge.update_symbolic_decision_outcome(decision_info, reward)

            # Record symbolic feedback in neural brain
            if decision_info.get('decision_type') == 'symbolic':
                success_rate = 1.0 if reward > 0 else 0.0
                skill_used = decision_info.get('reasoning', '')
                self.brain.record_symbolic_feedback('symbolic', success_rate, skill_used)

        # Periodic neural-symbolic synchronization
        if self.brain.env_training_steps % 20 == 0:
            self._synchronize_neural_symbolic()

    def _synchronize_neural_symbolic(self):
        """Synchronize neural and symbolic brains"""
        try:
            # Get latest neural insights
            neural_insights = self.brain.get_neural_insights_for_symbolic_brain()

            # Update symbolic knowledge with new insights
            self.knowledge.integrate_neural_insights(neural_insights)

            # Check for new transferable patterns
            transfer_readiness = neural_insights.get('transfer_readiness', 0.0)
            if transfer_readiness > 0.7:
                self._extract_transferable_patterns(neural_insights)

            print(f"🔄 Neural-symbolic synchronization completed (readiness: {transfer_readiness:.2f})")

        except Exception as e:
            print(f"❌ Error in neural-symbolic synchronization: {e}")

    def _extract_transferable_patterns(self, neural_insights: Dict[str, Any]):
        """Extract new transferable patterns from neural learning"""
        try:
            neural_skills = neural_insights.get('neural_skill_development', {})

            for skill, skill_data in neural_skills.items():
                confidence = skill_data.get('best_confidence', 0.0)

                if confidence > 0.8:  # High confidence threshold
                    # Create transferable pattern
                    pattern = {
                        'skill': skill,
                        'confidence': confidence,
                        'source_environment': self.environment_id,
                        'neural_derived': True,
                        'created_at': time.time(),
                        'ready_for_transfer': True
                    }

                    # Record transfer event
                    transfer_event = {
                        'type': 'pattern_extraction',
                        'pattern': pattern,
                        'environment': self.environment_id,
                        'timestamp': time.time()
                    }

                    self.session_transfer_events.append(transfer_event)

                    print(f"🎯 Transferable pattern extracted: {skill} (confidence: {confidence:.2f})")

        except Exception as e:
            print(f"❌ Error extracting transferable patterns: {e}")

    def end_enhanced_session(self, win: bool, reward: float, outcome_summary: str,
                             adaptive_params_used: Optional[Dict] = None):
        """End enhanced session with neural-symbolic analysis"""
        if self.current_app:
            # Standard session ending
            self.knowledge.record_game_result(self.current_app, win, reward, outcome_summary)

            # Enhanced neural-symbolic analysis
            session_analysis = self._analyze_session_performance(win, reward, outcome_summary)

            # Record enhanced transfer learning events
            for event in self.session_transfer_events:
                if win and reward > 0:  # Successful session
                    success_rate = min(1.0, (reward + 5) / 10)
                    self.brain.record_knowledge_transfer(
                        event.get('environment', self.environment_id),
                        event.get('type', 'unknown'),
                        success_rate
                    )

            # Generate session insights
            session_insights = self._generate_session_insights(session_analysis)

            print(f"✅ Enhanced session ended for {self.current_app}")
            print(f"   🔄 Transfer events: {len(self.session_transfer_events)}")
            print(f"   🧠🧩 Neural-symbolic decisions: {len(self.decision_correlation_history)}")
            print(f"   📊 Session analysis: {session_analysis['integration_effectiveness']:.2f}")

        self.current_app = None
        self.current_context = None
        self.session_transfer_events = []

    def _analyze_session_performance(self, win: bool, reward: float, outcome_summary: str) -> Dict[str, Any]:
        """Analyze session performance with neural-symbolic metrics"""
        analysis = {
            'session_success': win,
            'session_reward': reward,
            'outcome_summary': outcome_summary,
            'neural_symbolic_decisions': len([d for d in self.decision_correlation_history
                                              if d.get('decision_type') == 'symbolic']),
            'neural_decisions': len([d for d in self.decision_correlation_history
                                     if d.get('decision_type') == 'neural']),
            'decision_agreement_rate': self._calculate_decision_agreement(),
            'integration_effectiveness': self._calculate_session_integration_effectiveness(),
            'transferable_patterns_identified': len(self.session_transfer_events)
        }

        return analysis

    def _calculate_decision_agreement(self) -> float:
        """Calculate agreement rate between neural and symbolic decisions"""
        if not self.decision_correlation_history:
            return 0.0

        agreements = 0
        total_decisions = len(self.decision_correlation_history)

        for decision in self.decision_correlation_history:
            if decision.get('neural_action') == decision.get('symbolic_action'):
                agreements += 1

        return agreements / total_decisions if total_decisions > 0 else 0.0

    def _calculate_session_integration_effectiveness(self) -> float:
        """Calculate effectiveness of neural-symbolic integration in this session"""
        if not self.integration_history:
            return 0.0

        recent_integration = self.integration_history[-1] if self.integration_history else {}
        return recent_integration.get('integration_quality', 0.0)

    def _generate_session_insights(self, session_analysis: Dict[str, Any]) -> List[str]:
        """Generate insights about neural-symbolic performance"""
        insights = []

        decision_agreement = session_analysis.get('decision_agreement_rate', 0.0)
        if decision_agreement > 0.8:
            insights.append("High neural-symbolic coherence - both brains aligned")
        elif decision_agreement < 0.4:
            insights.append("Low neural-symbolic coherence - brains learning different patterns")

        integration_effectiveness = session_analysis.get('integration_effectiveness', 0.0)
        if integration_effectiveness > 0.7:
            insights.append("Strong neural-symbolic integration - effective knowledge transfer")
        elif integration_effectiveness < 0.3:
            insights.append("Weak neural-symbolic integration - more synchronization needed")

        transferable_patterns = session_analysis.get('transferable_patterns_identified', 0)
        if transferable_patterns > 2:
            insights.append(f"High transfer learning potential - {transferable_patterns} patterns identified")

        return insights

    def save_enhanced_all(self):
        """Save both enhanced brain and knowledge with neural-symbolic integration"""
        brain_saved = self.brain.save_brain()
        knowledge_saved = self.knowledge.save_knowledge()

        if brain_saved and knowledge_saved:
            print(f"💾 Enhanced Dual Brain saved with neural-symbolic integration")
            print(
                f"   🧠 Neural Brain: {self.brain.env_training_steps} env steps, {self.brain.total_training_steps} total")
            print(f"   🧩 Symbolic Knowledge: Neural-symbolic integration active")
            print(f"   🔗 Integration History: {len(self.integration_history)} events")
            print(f"   🔄 Decision Correlations: {len(self.decision_correlation_history)} tracked")

        return brain_saved and knowledge_saved

    def get_enhanced_transfer_learning_report(self) -> Dict[str, Any]:
        """Get comprehensive enhanced transfer learning report"""
        brain_insights = self.brain.get_neural_insights_for_symbolic_brain()
        knowledge_summary = self.knowledge.get_neural_symbolic_integration_report()

        return {
            'agent_id': self.agent_id,
            'current_environment': self.environment_id,
            'architecture_version': self.brain.architecture_version,

            # Neural brain insights
            'neural_brain': {
                'total_training_steps': self.brain.total_training_steps,
                'neural_skill_development': brain_insights.get('neural_skill_development', {}),
                'transfer_readiness': brain_insights.get('transfer_readiness', 0.0),
                'learning_trajectory': brain_insights.get('learning_trajectory', {}),
                'meta_learning_state': brain_insights.get('meta_learning_state', {})
            },

            # Symbolic knowledge insights
            'symbolic_knowledge': knowledge_summary,

            # Integration analysis
            'neural_symbolic_integration': {
                'integration_active': self.neural_symbolic_integration_active,
                'integration_events': len(self.integration_history),
                'decision_correlation_rate': self._calculate_decision_agreement(),
                'integration_effectiveness': self._calculate_session_integration_effectiveness(),
                'recent_insights': self.integration_history[-3:] if self.integration_history else []
            },

            # Transfer learning summary
            'transfer_learning': {
                'total_environments': len(self.knowledge.knowledge.get('environment_specific', {})),
                'transferable_patterns_available': len(self.session_transfer_events),
                'cross_environment_ready': knowledge_summary.get('transfer_readiness', 0.0) > 0.6,
                'neural_symbolic_coherence': brain_insights.get('meta_learning_state', {}).get(
                    'neural_symbolic_coherence', 0.0)
            }
        }


# Example integration and testing
if __name__ == "__main__":
    print("🧪 Testing Enhanced Dual Brain System with Neural-Symbolic Integration...")

    # Create enhanced dual brain agent
    agent = EnhancedDualBrainAgent(
        agent_id="test_agent_neural_symbolic_001",
        environment_id="pong"
    )

    # Test enhanced environment context
    test_env_context = {
        'name': 'pong',
        'objective': {'primary': 'Score 21 points using neural-symbolic intelligence'},
        'strategic_concepts': {
            'core_skills': ['Neural trajectory prediction', 'Symbolic strategy adaptation'],
            'tactical_approaches': ['Neural-symbolic decision fusion'],
            'success_patterns': ['Coherent neural-symbolic alignment']
        },
        'learning_recommendations': {
            'neural_focus': ['Pattern extraction for symbolic interpretation'],
            'symbolic_focus': ['Strategy guidance for neural learning'],
            'integration_focus': ['Neural-symbolic coherence optimization']
        }
    }

    # Start enhanced session
    context = agent.start_enhanced_session("pong", env_context=test_env_context)

    # Simulate enhanced learning with neural-symbolic integration
    for i in range(50):
        # Mock neural state and Q-values
        neural_state = np.random.random(256) * 2 - 1
        neural_q_values = np.random.random(3)

        # Enhanced decision making every 5 steps
        if i % 5 == 0:
            action, reasoning, decision_info = agent.make_enhanced_decision(
                neural_state, neural_q_values, exploration_rate=0.3
            )

            print(f"Step {i}: {reasoning}")
            if decision_info.get('decision_type') == 'symbolic':
                print(f"   🧩 Symbolic decision with neural insights")

            # Record enhanced learning step
            mock_reward = random.uniform(-1, 5)
            mock_loss = random.uniform(0, 1)

            agent.record_enhanced_learning_step(
                mock_reward, mock_loss, 0.3, neural_state, action, decision_info
            )

    # End enhanced session
    agent.end_enhanced_session(True, 8.5, "Won with excellent neural-symbolic coordination")

    # Get enhanced transfer learning report
    enhanced_report = agent.get_enhanced_transfer_learning_report()

    print(f"\n📊 ENHANCED TRANSFER LEARNING REPORT:")
    print(f"   Agent ID: {enhanced_report['agent_id']}")
    print(f"   Architecture: {enhanced_report['architecture_version']}")
    print(f"   Neural Training Steps: {enhanced_report['neural_brain']['total_training_steps']}")
    print(f"   Neural Transfer Readiness: {enhanced_report['neural_brain']['transfer_readiness']:.2f}")
    print(f"   Symbolic Integration Quality: {enhanced_report['symbolic_knowledge'].get('integration_quality', 0):.2f}")
    print(f"   Neural-Symbolic Coherence: {enhanced_report['transfer_learning']['neural_symbolic_coherence']:.2f}")
    print(
        f"   Decision Correlation Rate: {enhanced_report['neural_symbolic_integration']['decision_correlation_rate']:.2f}")

    # Save enhanced system
    agent.save_enhanced_all()

    print(f"\n✅ Enhanced Dual Brain System with Neural-Symbolic Integration test complete!")
    print(f"🧠🧩 Neural and Symbolic brains now work together as integrated intelligence!")
    print(f"🔗 Neural patterns are interpreted symbolically for true understanding!")
    print(f"🌍 Ready for universal transfer learning across any environment!")