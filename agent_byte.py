# agent_byte.py - Enhanced Modular Transferable AI Agent with Neural-Symbolic Integration
import numpy as np
import json
import time
import random
import os
import shutil
from collections import deque
import datetime
from typing import Dict, Optional, List, Any, Tuple
import uuid

# Import the enhanced dual brain system with neural-symbolic integration
from dual_brain_system import EnhancedDualBrainAgent, EnhancedAgentBrain, EnhancedAgentKnowledge
# Import the decision agent for expert selection
from decision_agent import DecisionAgent, USE_RULE, USE_DDQN


class StandardizedNetwork:
    """
    Standardized Neural Network Architecture for Transfer Learning

    Enhanced with better neural pattern tracking for symbolic interpretation
    """

    def __init__(self, environment_id: str, action_size: int, learning_rate=0.001):
        self.environment_id = environment_id
        self.action_size = action_size
        self.learning_rate = learning_rate

        # STANDARDIZED ARCHITECTURE - 256 input for all environments
        self.input_size = 256
        self.core_sizes = [512, 256, 128]  # Transferable core layers
        self.adapter_size = 64  # Environment adaptation layer

        # Initialize network layers
        self._initialize_core_layers()
        self._initialize_adapter_layer()
        self._initialize_output_layer()

        # Enhanced metadata for neural-symbolic integration
        self.metadata = {
            'environment_id': environment_id,
            'architecture_version': 'Agent Byte v2.1 - Neural-Symbolic Integration',
            'created_at': time.time(),
            'training_domains': [environment_id],
            'skill_tags': [],
            'transfer_readiness': 0.0,
            'core_layers_frozen': False,
            'transfer_source': None,
            'neural_symbolic_compatible': True,
            'pattern_tracking_enabled': True
        }

        # Pattern tracking for symbolic interpretation
        self.activation_patterns = []
        self.decision_patterns = []
        self.reward_correlations = {}

        print(f"🧠 Enhanced Standardized Network initialized for {environment_id}")
        print(f"   Architecture: {self.input_size}→{self.core_sizes}→{self.adapter_size}→{self.action_size}")
        print(f"   Neural-Symbolic Integration: Ready")

    def _initialize_core_layers(self):
        """Initialize transferable core feature layers with pattern tracking"""
        self.core_layers = []

        layer_sizes = [self.input_size] + self.core_sizes
        for i in range(len(layer_sizes) - 1):
            layer = {
                'weights': np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i]),
                'biases': np.zeros(layer_sizes[i + 1]),
                'layer_type': 'core',
                'transferable': True,
                'learning_history': [],
                'activation_patterns': [],  # NEW: Track activation patterns
                'pattern_significance': 0.0  # NEW: Track pattern importance
            }
            self.core_layers.append(layer)

        print(
            f"   ✅ Enhanced core layers initialized: {len(self.core_layers)} transferable layers with pattern tracking")

    def _initialize_adapter_layer(self):
        """Initialize environment-specific adapter layer"""
        core_output_size = self.core_sizes[-1]

        self.adapter_layer = {
            'weights': np.random.randn(core_output_size, self.adapter_size) * np.sqrt(2.0 / core_output_size),
            'biases': np.zeros(self.adapter_size),
            'layer_type': 'adapter',
            'transferable': False,
            'environment_specific': True
        }

        print(f"   🔧 Enhanced adapter layer initialized: {core_output_size}→{self.adapter_size}")

    def _initialize_output_layer(self):
        """Initialize environment-specific output layer"""
        self.output_layer = {
            'weights': np.random.randn(self.adapter_size, self.action_size) * np.sqrt(2.0 / self.adapter_size),
            'biases': np.zeros(self.action_size),
            'layer_type': 'output',
            'transferable': False,
            'environment_specific': True
        }

        print(f"   🎯 Enhanced output layer initialized: {self.adapter_size}→{self.action_size}")

    def normalize_input(self, raw_state: np.ndarray) -> np.ndarray:
        """Enhanced normalize any environment state to standard 256-dimension input"""
        if len(raw_state) >= 256:
            # Truncate if too large
            normalized = raw_state[:256]
        else:
            # Pad with zeros if too small, and add enhanced positional encoding
            normalized = np.zeros(256)
            normalized[:len(raw_state)] = raw_state

            # Enhanced positional encoding for better pattern recognition
            for i in range(len(raw_state), 256):
                # More sophisticated encoding for symbolic interpretation
                normalized[i] = np.sin(i * 0.01) * 0.1 + np.cos(i * 0.005) * 0.05

        # Ensure all values are in reasonable range
        normalized = np.clip(normalized, -10, 10)

        return normalized

    def leaky_relu(self, x, alpha=0.01):
        """Leaky ReLU activation function"""
        return np.where(x > 0, x, alpha * x)

    def forward(self, raw_state: np.ndarray) -> np.ndarray:
        """Enhanced forward pass with pattern tracking for symbolic interpretation"""
        # Normalize input to standard size
        state = self.normalize_input(raw_state)

        # Forward through core layers (transferable features)
        x = state.copy()
        self.core_activations = [x]

        for i, layer in enumerate(self.core_layers):
            z = np.dot(x, layer['weights']) + layer['biases']
            x = self.leaky_relu(z)
            self.core_activations.append(x)

            # Track activation patterns for symbolic interpretation
            if len(self.activation_patterns) < 100:  # Keep recent patterns
                pattern = {
                    'layer': i,
                    'activation_mean': np.mean(x),
                    'activation_std': np.std(x),
                    'activation_sparsity': np.sum(x == 0) / len(x),
                    'timestamp': time.time()
                }
                self.activation_patterns.append(pattern)

        # Store enhanced core features for transfer learning
        self.core_features = x.copy()

        # Forward through adapter layer (environment-specific)
        adapter_z = np.dot(x, self.adapter_layer['weights']) + self.adapter_layer['biases']
        adapter_out = self.leaky_relu(adapter_z)
        self.adapter_features = adapter_out.copy()

        # Forward through output layer
        output_z = np.dot(adapter_out, self.output_layer['weights']) + self.output_layer['biases']
        q_values = output_z  # Linear output for Q-learning

        return q_values

    def record_decision_pattern(self, state: np.ndarray, action: int, reward: float):
        """Record decision patterns for symbolic interpretation"""
        decision_pattern = {
            'state_summary': np.array([np.mean(state), np.std(state), np.min(state), np.max(state)]),
            'action': action,
            'reward': reward,
            'core_features_summary': np.array([np.mean(self.core_features), np.std(self.core_features)]) if hasattr(
                self, 'core_features') else np.zeros(2),
            'timestamp': time.time()
        }

        self.decision_patterns.append(decision_pattern)

        # Keep manageable history
        if len(self.decision_patterns) > 200:
            self.decision_patterns = self.decision_patterns[-100:]

        # Update reward correlations
        if action not in self.reward_correlations:
            self.reward_correlations[action] = []
        self.reward_correlations[action].append(reward)

        # Keep recent correlations
        if len(self.reward_correlations[action]) > 50:
            self.reward_correlations[action] = self.reward_correlations[action][-25:]

    def get_enhanced_core_features(self) -> Dict[str, Any]:
        """Get enhanced transferable core features with pattern analysis"""
        core_features = getattr(self, 'core_features', np.zeros(self.core_sizes[-1]))

        return {
            'core_features': core_features,
            'feature_summary': {
                'mean': np.mean(core_features),
                'std': np.std(core_features),
                'sparsity': np.sum(core_features == 0) / len(core_features),
                'max_activation': np.max(core_features),
                'min_activation': np.min(core_features)
            },
            'recent_patterns': self.activation_patterns[-10:] if self.activation_patterns else [],
            'decision_patterns': self.decision_patterns[-10:] if self.decision_patterns else [],
            'reward_correlations': {k: np.mean(v) for k, v in self.reward_correlations.items()},
            'pattern_stability': self._calculate_pattern_stability()
        }

    def _calculate_pattern_stability(self) -> float:
        """Calculate stability of neural patterns for transfer learning"""
        if len(self.activation_patterns) < 10:
            return 0.0

        recent_patterns = self.activation_patterns[-10:]
        means = [p['activation_mean'] for p in recent_patterns]
        stds = [p['activation_std'] for p in recent_patterns]

        mean_stability = 1.0 - (np.std(means) / (np.mean(np.abs(means)) + 1e-8))
        std_stability = 1.0 - (np.std(stds) / (np.mean(stds) + 1e-8))

        return max(0, min(1, (mean_stability + std_stability) / 2))

    def transfer_core_layers_from(self, source_network: 'StandardizedNetwork'):
        """Transfer core layers from another network"""
        try:
            # Verify architecture compatibility
            if self.core_sizes != source_network.core_sizes:
                print(f"❌ Incompatible core architectures")
                return False

            # Copy core layer weights
            for i, source_layer in enumerate(source_network.core_layers):
                self.core_layers[i]['weights'] = source_layer['weights'].copy()
                self.core_layers[i]['biases'] = source_layer['biases'].copy()

            # Update metadata
            source_domains = source_network.metadata.get('training_domains', [])
            self.metadata['training_domains'].extend(source_domains)
            self.metadata['training_domains'] = list(set(self.metadata['training_domains']))
            self.metadata['transfer_source'] = source_network.environment_id

            print(f"🔄 Core layers transferred from {source_network.environment_id}")
            print(f"   Combined training domains: {self.metadata['training_domains']}")
            return True

        except Exception as e:
            print(f"❌ Error transferring core layers: {e}")
            return False

    def save_network(self, filepath: str):
        """Save enhanced network with pattern data"""
        try:
            # Prepare enhanced data for saving
            save_data = {
                'metadata': self.metadata,
                'architecture': {
                    'input_size': self.input_size,
                    'core_sizes': self.core_sizes,
                    'adapter_size': self.adapter_size,
                    'action_size': self.action_size
                },
                'pattern_data': {
                    'activation_patterns': self.activation_patterns[-50:],  # Recent patterns
                    'decision_patterns': self.decision_patterns[-50:],
                    'reward_correlations': self.reward_correlations
                }
            }

            # Save core layers
            for i, layer in enumerate(self.core_layers):
                save_data[f'core_layer_{i}_weights'] = layer['weights']
                save_data[f'core_layer_{i}_biases'] = layer['biases']

            # Save adapter and output layers
            save_data['adapter_weights'] = self.adapter_layer['weights']
            save_data['adapter_biases'] = self.adapter_layer['biases']
            save_data['output_weights'] = self.output_layer['weights']
            save_data['output_biases'] = self.output_layer['biases']

            # Save to file
            np.savez_compressed(filepath, **save_data)

            print(f"💾 Enhanced network saved: {filepath}")
            print(f"   Pattern data: {len(self.activation_patterns)} activation patterns")
            return True

        except Exception as e:
            print(f"❌ Error saving enhanced network: {e}")
            return False

    def load_network(self, filepath: str) -> bool:
        """Load enhanced network with pattern data"""
        try:
            if not os.path.exists(filepath):
                print(f"⚠️ Enhanced network file not found: {filepath}")
                return False

            # Load data
            data = np.load(filepath, allow_pickle=True)

            # Load core layers
            for i in range(len(self.core_layers)):
                self.core_layers[i]['weights'] = data[f'core_layer_{i}_weights']
                self.core_layers[i]['biases'] = data[f'core_layer_{i}_biases']

            # Load adapter and output layers
            self.adapter_layer['weights'] = data['adapter_weights']
            self.adapter_layer['biases'] = data['adapter_biases']
            self.output_layer['weights'] = data['output_weights']
            self.output_layer['biases'] = data['output_biases']

            # Load enhanced pattern data
            if 'pattern_data' in data:
                pattern_data = data['pattern_data'].item()
                self.activation_patterns = pattern_data.get('activation_patterns', [])
                self.decision_patterns = pattern_data.get('decision_patterns', [])
                self.reward_correlations = pattern_data.get('reward_correlations', {})

            # Update metadata
            if 'metadata' in data:
                self.metadata.update(data['metadata'].item())

            print(f"📥 Enhanced network loaded: {filepath}")
            print(f"   Pattern data: {len(self.activation_patterns)} activation patterns")
            return True

        except Exception as e:
            print(f"❌ Error loading enhanced network: {e}")
            return False


class EnhancedMatchLogger:
    """Enhanced Match logging system with neural-symbolic integration tracking"""

    def __init__(self, agent_id: str, environment_id: str):
        self.agent_id = agent_id
        self.environment_id = environment_id
        self.log_dir = f"saas_agents/{agent_id}/environments/{environment_id}"
        self.log_file = os.path.join(self.log_dir, 'matches.json')
        self.current_match = None
        self.matches = []

        # Ensure directory structure exists
        os.makedirs(self.log_dir, exist_ok=True)

        self.load_match_history()

    def start_match(self, match_id: str):
        """Start logging an enhanced match with neural-symbolic tracking"""
        self.current_match = {
            'match_id': match_id,
            'environment_id': self.environment_id,
            'agent_id': self.agent_id,
            'start_time': datetime.datetime.now().isoformat(),
            'end_time': None,
            'winner': None,
            'final_score': {},

            # Enhanced neural-symbolic tracking
            'neural_patterns_tracked': 0,
            'symbolic_decisions_made': 0,
            'neural_symbolic_coherence': 0.0,
            'transferable_skills_used': [],
            'environment_specific_tactics': [],
            'learning_insights': [],
            'neural_pattern_evolution': [],
            'knowledge_transfer_events': [],
            'decision_correlation_analysis': []
        }
        print(f"🆕 Started enhanced match logging: {match_id}")

    def log_neural_symbolic_decision(self, decision_info: Dict[str, Any]):
        """Log neural-symbolic decision event"""
        if self.current_match:
            decision_event = {
                'timestamp': time.time(),
                'decision_type': decision_info.get('decision_type'),
                'neural_action': decision_info.get('neural_action'),
                'symbolic_action': decision_info.get('symbolic_action'),
                'reasoning': decision_info.get('reasoning'),
                'confidence': decision_info.get('confidence', 0.0)
            }
            self.current_match['decision_correlation_analysis'].append(decision_event)

            if decision_info.get('decision_type') == 'symbolic':
                self.current_match['symbolic_decisions_made'] += 1

    def log_neural_pattern_evolution(self, pattern_info: Dict[str, Any]):
        """Log neural pattern evolution"""
        if self.current_match:
            pattern_event = {
                'timestamp': time.time(),
                'pattern_stability': pattern_info.get('pattern_stability', 0.0),
                'core_features_summary': pattern_info.get('feature_summary', {}),
                'reward_correlations': pattern_info.get('reward_correlations', {}),
                'transfer_readiness': pattern_info.get('transfer_readiness', 0.0)
            }
            self.current_match['neural_pattern_evolution'].append(pattern_event)
            self.current_match['neural_patterns_tracked'] += 1

    def end_match(self, winner: str, final_scores: Dict, enhanced_stats: Dict):
        """End enhanced match logging with neural-symbolic insights"""
        if not self.current_match:
            return

        try:
            self.current_match['end_time'] = datetime.datetime.now().isoformat()
            self.current_match['winner'] = winner
            self.current_match['final_score'] = final_scores

            # Add enhanced neural-symbolic insights
            if enhanced_stats:
                self.current_match['neural_symbolic_insights'] = {
                    'pattern_stability': enhanced_stats.get('pattern_stability', 0.0),
                    'decision_coherence': enhanced_stats.get('decision_coherence', 0.0),
                    'transfer_effectiveness': enhanced_stats.get('transfer_effectiveness', 0.0),
                    'learning_acceleration': enhanced_stats.get('learning_acceleration', 0.0)
                }

            # Calculate enhanced metrics
            self._calculate_enhanced_match_metrics()

            # Add to matches and save
            self.matches.append(self.current_match.copy())
            self.save_match_history()

            print(f"📊 Enhanced match {self.current_match['match_id']} completed")
            print(f"   Neural patterns: {self.current_match['neural_patterns_tracked']}")
            print(f"   Symbolic decisions: {self.current_match['symbolic_decisions_made']}")

            self.current_match = None

        except Exception as e:
            print(f"❌ Error ending enhanced match: {e}")
            self.current_match = None

    def _calculate_enhanced_match_metrics(self):
        """Calculate enhanced match metrics"""
        try:
            # Calculate neural-symbolic coherence
            decisions = self.current_match.get('decision_correlation_analysis', [])
            if decisions:
                coherence_scores = []
                for decision in decisions:
                    if decision.get('neural_action') == decision.get('symbolic_action'):
                        coherence_scores.append(1.0)
                    else:
                        coherence_scores.append(0.0)

                self.current_match['neural_symbolic_coherence'] = np.mean(coherence_scores) if coherence_scores else 0.0

            # Calculate pattern evolution trajectory
            patterns = self.current_match.get('neural_pattern_evolution', [])
            if len(patterns) >= 2:
                early_stability = np.mean([p.get('pattern_stability', 0) for p in patterns[:len(patterns) // 2]])
                late_stability = np.mean([p.get('pattern_stability', 0) for p in patterns[len(patterns) // 2:]])
                self.current_match['pattern_improvement'] = late_stability - early_stability

        except Exception as e:
            print(f"❌ Error calculating enhanced metrics: {e}")

    def load_match_history(self):
        """Load enhanced match history"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    self.matches = data.get('matches', [])
                    # Keep only recent matches (last 20)
                    if len(self.matches) > 20:
                        self.matches = self.matches[-20:]
                print(f"📚 Loaded {len(self.matches)} enhanced match records")
        except Exception as e:
            print(f"⚠️ Could not load enhanced match history: {e}")
            self.matches = []

    def save_match_history(self):
        """Save enhanced match history"""
        try:
            # Keep only recent matches
            if len(self.matches) > 20:
                self.matches = self.matches[-20:]

            data = {
                'agent_id': self.agent_id,
                'environment_id': self.environment_id,
                'total_matches': len(self.matches),
                'last_updated': datetime.datetime.now().isoformat(),
                'architecture_version': 'Agent Byte v2.1 - Neural-Symbolic Integration',
                'enhanced_features': {
                    'neural_symbolic_tracking': True,
                    'pattern_evolution_tracking': True,
                    'transfer_learning_analytics': True
                },
                'matches': self.matches
            }

            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"❌ Could not save enhanced match history: {e}")


class EnhancedAgentByte:
    """Enhanced Modular Agent Byte with Neural-Symbolic Dual Brain Integration"""

    def __init__(self, agent_id: str, environment_id: str, raw_state_size: int, action_size: int):
        print("🚀 Agent Byte v2.1 - Enhanced Neural-Symbolic Integration Initializing...")

        # Core identification
        self.agent_id = agent_id
        self.environment_id = environment_id
        self.raw_state_size = raw_state_size
        self.action_size = action_size

        # Initialize enhanced file structure
        self._initialize_enhanced_agent_structure()

        # Initialize enhanced dual brain system with neural-symbolic integration
        self.dual_brain = EnhancedDualBrainAgent(
            agent_id=agent_id,
            environment_id=environment_id,
            brain_file=self.brain_file,
            knowledge_file=self.knowledge_file,
            canonical_knowledge_file=self.canonical_knowledge_file if hasattr(self, 'canonical_knowledge_file') else None,
            default_knowledge_file=self.default_environment_knowledge_file
        )

        # Initialize enhanced standardized neural network with pattern tracking
        self.network = StandardizedNetwork(environment_id, action_size)
        self.target_network = StandardizedNetwork(environment_id, action_size)
        self.target_network.transfer_core_layers_from(self.network)

        # Load existing networks if available
        self._load_or_initialize_enhanced_networks()

        # Learning parameters
        self.learning_rate = 0.001
        self.exploration_rate = 0.8
        self.exploration_decay = 0.995
        self.min_exploration = 0.1
        self.gamma = 0.99
        self.target_update_frequency = 1000

        # Enhanced experience and demo buffers
        self.experience_buffer = deque(maxlen=5000)
        self.user_demo_buffer = deque(maxlen=1000)
        self.replay_batch_size = 16

        # Enhanced performance tracking with neural-symbolic metrics
        self.games_played = 0
        self.wins = 0
        self.total_reward = 0
        self.match_reward = 0
        self.actions_taken = 0
        self.training_steps = 0

        # Enhanced transfer learning tracking with neural-symbolic integration
        self.transferable_skills_used = []
        self.knowledge_transfer_events = []
        self.cross_environment_insights = []
        self.neural_symbolic_decisions = []
        self.pattern_evolution_history = []

        # Environment integration
        self.env = None
        self.env_context = None
        self.app_context = None

        # Enhanced logger with neural-symbolic tracking
        self.logger = EnhancedMatchLogger(agent_id, environment_id)

        # Initialize Decision Agent for expert selection
        # Use normalized state size (256) since network normalizes inputs
        self.decision_agent = DecisionAgent(
            state_size=256,  # StandardizedNetwork normalizes to 256 dimensions
            learning_rate=0.001,
            gamma=self.gamma
        )
        
        # Load existing decision agent if available
        self._load_decision_agent()
        
        # Track last state/actions for decision agent logging
        self.last_state = None
        self.last_ddqn_action = None
        self.last_rule_action = None
        self.last_expert_choice = None

        print("✅ Enhanced Agent Byte v2.1 Created with Neural-Symbolic Integration!")
        print(f"   🆔 Agent ID: {agent_id}")
        print(f"   🌍 Environment: {environment_id}")
        print(f"   🧠 Enhanced Standardized Network: 256→512→256→128→64→{action_size}")
        print(f"   🧠🧩 Enhanced Dual Brain: Neural-Symbolic Integration Active")
        print(f"   🎯 Decision Agent: Expert Selection Active")
        print(f"   🔄 Transfer Learning: Cross-environment ready")
        print(f"   📁 Enhanced Storage: {self.agent_dir}")

    def _initialize_enhanced_agent_structure(self):
        """Initialize enhanced multi-environment file structure"""
        # Create enhanced agent directory structure
        self.agent_dir = f"saas_agents/{self.agent_id}"
        self.core_dir = os.path.join(self.agent_dir, "core")
        self.env_dir = os.path.join(self.agent_dir, "environments", self.environment_id)
        self.transfer_dir = os.path.join(self.agent_dir, "transfers")
        self.neural_symbolic_dir = os.path.join(self.agent_dir, "neural_symbolic")  # NEW

        # Create directories
        for directory in [self.core_dir, self.env_dir, self.transfer_dir, self.neural_symbolic_dir]:
            os.makedirs(directory, exist_ok=True)

        # Define enhanced file paths
        self.profile_file = os.path.join(self.core_dir, "agent_profile.json")
        self.general_knowledge_file = os.path.join(self.core_dir, "general_knowledge.json")
        self.canonical_knowledge_file = os.path.join(self.core_dir, "knowledge.json")
        self.meta_learning_file = os.path.join(self.core_dir, "meta_learning.json")

        # Neural-symbolic integration files
        self.neural_patterns_file = os.path.join(self.neural_symbolic_dir, "neural_patterns.json")
        self.decision_correlations_file = os.path.join(self.neural_symbolic_dir, "decision_correlations.json")

        self.network_file = os.path.join(self.env_dir, "network.npz")
        self.brain_file = os.path.join(self.env_dir, "brain.json")
        self.personalized_knowledge_file = os.path.join(
            self.env_dir, f"{self.environment_id}_knowledge.json")
        self.default_environment_knowledge_file = os.path.join(
            self.env_dir, f"default_{self.environment_id}_knowledge.json")
        self.legacy_knowledge_file = os.path.join(self.env_dir, "knowledge.json")
        self.knowledge_file = self.personalized_knowledge_file
        self.decision_agent_file = os.path.join(self.env_dir, "decision_agent.pkl")

        self._migrate_environment_knowledge_files()

        # Initialize enhanced agent profile
        self._initialize_enhanced_agent_profile()

        print(f"📁 Enhanced agent structure initialized: {self.agent_dir}")

    def _migrate_environment_knowledge_files(self):
        """Ensure personalized/default knowledge files exist and migrate legacy data."""
        try:
            env_dir_exists = os.path.exists(self.env_dir)
            if not env_dir_exists:
                os.makedirs(self.env_dir, exist_ok=True)

            # Legacy knowledge.json migration
            if os.path.exists(self.legacy_knowledge_file):
                if not os.path.exists(self.default_environment_knowledge_file):
                    shutil.copy2(self.legacy_knowledge_file, self.default_environment_knowledge_file)
                if not os.path.exists(self.personalized_knowledge_file):
                    shutil.copy2(self.legacy_knowledge_file, self.personalized_knowledge_file)
                try:
                    os.remove(self.legacy_knowledge_file)
                except OSError:
                    pass

            # If personalized file missing but default exists, clone default
            if (not os.path.exists(self.personalized_knowledge_file)
                    and os.path.exists(self.default_environment_knowledge_file)):
                shutil.copy2(self.default_environment_knowledge_file, self.personalized_knowledge_file)

            # If default missing but personalized exists, seed default from personalized
            if os.path.exists(self.personalized_knowledge_file) and not os.path.exists(
                    self.default_environment_knowledge_file):
                shutil.copy2(self.personalized_knowledge_file, self.default_environment_knowledge_file)

            if not os.path.exists(self.personalized_knowledge_file):
                self._write_empty_environment_knowledge(self.personalized_knowledge_file)

            if not os.path.exists(self.default_environment_knowledge_file):
                self._write_empty_environment_knowledge(self.default_environment_knowledge_file)

        except Exception as e:
            print(f"⚠️ Knowledge migration warning: {e}")

    def _write_empty_environment_knowledge(self, filepath: str):
        """Create a minimal environment knowledge file as a placeholder."""
        try:
            knowledge = {
                "general_knowledge": {
                    "transferable_strategies": [],
                    "meta_learning_principles": [],
                    "cross_environment_patterns": [],
                    "abstract_concepts": [],
                    "neural_symbolic_correlations": []
                },
                "environment_specific": {
                    self.environment_id: {
                        "environment_profile": {
                            "environment_id": self.environment_id,
                            "display_name": self.environment_id.replace('_', ' ').title(),
                            "environment_type": "unknown",
                            "understanding_level": "basic",
                            "total_sessions": 0,
                            "first_encountered": time.time(),
                            "last_updated": time.time()
                        },
                        "objectives": {},
                        "rules": {},
                        "strategic_framework": {},
                        "strategies": [],
                        "lessons": [],
                        "tactical_knowledge": [],
                        "performance_patterns": [],
                        "neural_insights": [],
                        "knowledge_unlocks": [],
                        "experiment_logs": []
                    }
                },
                "transfer_mappings": {
                    "strategy_abstractions": {},
                    "concept_translations": {},
                    "success_patterns": [],
                    "neural_pattern_mappings": {}
                },
                "symbolic_decision_history": [],
                "metadata": {
                    "version": "2.2.0 - Personalized Environment Knowledge",
                    "agent_id": self.agent_id,
                    "environments": [self.environment_id],
                    "created": datetime.datetime.now().isoformat(),
                    "last_updated": datetime.datetime.now().isoformat(),
                    "transfer_learning_enabled": True,
                    "neural_symbolic_integration": True,
                    "environment_knowledge_integration": True
                }
            }

            with open(filepath, 'w') as f:
                json.dump(knowledge, f, indent=2)

        except Exception as e:
            print(f"⚠️ Could not create placeholder knowledge file {filepath}: {e}")

    def _initialize_enhanced_agent_profile(self):
        """Initialize or load enhanced agent profile with neural-symbolic tracking"""
        if os.path.exists(self.profile_file):
            with open(self.profile_file, 'r') as f:
                self.profile = json.load(f)
            print(f"📋 Loaded enhanced agent profile: {len(self.profile.get('environments', []))} environments")
        else:
            # Create new enhanced agent profile
            self.profile = {
                'agent_id': self.agent_id,
                'created_at': datetime.datetime.now().isoformat(),
                'architecture_version': 'Agent Byte v2.1 - Neural-Symbolic Integration',
                'environments': [self.environment_id],
                'enhanced_capabilities': {
                    'transferable_skills': [],
                    'specialized_tactics': {},
                    'learning_efficiency': 0.0,
                    'adaptation_speed': 0.0,
                    'neural_symbolic_coherence': 0.0,  # NEW
                    'pattern_recognition_ability': 0.0,  # NEW
                    'cross_environment_transfer_rate': 0.0  # NEW
                },
                'neural_symbolic_integration': {
                    'enabled': True,
                    'decision_correlation_rate': 0.0,
                    'pattern_stability_score': 0.0,
                    'transfer_effectiveness': 0.0
                },
                'transfer_history': [],
                'performance_benchmarks': {}
            }
            self._save_enhanced_agent_profile()
            print(f"🆕 Created new enhanced agent profile with neural-symbolic integration")

    def _save_enhanced_agent_profile(self):
        """Save enhanced agent profile"""
        try:
            with open(self.profile_file, 'w') as f:
                json.dump(self.profile, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving enhanced agent profile: {e}")

    def _load_or_initialize_enhanced_networks(self):
        """Load existing enhanced networks or initialize new ones"""
        if os.path.exists(self.network_file):
            if self.network.load_network(self.network_file):
                self.target_network.load_network(self.network_file)
                print(f"📥 Loaded existing enhanced networks for {self.environment_id}")
            else:
                print(f"⚠️ Failed to load enhanced networks, using new initialization")
        else:
            print(f"🆕 No existing enhanced networks found, using new initialization")
    
    def _load_decision_agent(self):
        """Load existing decision agent if available"""
        if os.path.exists(self.decision_agent_file):
            if self.decision_agent.load(self.decision_agent_file):
                print(f"📥 Loaded existing Decision Agent for {self.environment_id}")
            else:
                print(f"⚠️ Failed to load Decision Agent, using new initialization")
        else:
            print(f"🆕 No existing Decision Agent found, using new initialization")

    def set_environment(self, env):
        """Set the environment instance for modular behavior"""
        self.env = env
        if hasattr(env, 'get_env_context'):
            self.env_context = env.get_env_context()
            print(f"🌟 Environment context loaded: {self.env_context.get('name', 'unknown')}")

    def start_new_match(self, match_id: str, env_context: Optional[Dict] = None):
        """
        🚀 Enhanced match start with permanent environment knowledge integration
        
        This now ensures agents permanently learn and remember what each environment is about
        """
        # Reset match-specific stats
        self.match_reward = 0
        self.actions_taken = 0
        self.transferable_skills_used = []
        self.knowledge_transfer_events = []
        self.neural_symbolic_decisions = []
        self.pattern_evolution_history = []

        # 🧠 STEP 1: Load existing environment understanding
        print(f"🧠 Loading environment understanding for {self.environment_id}...")
        existing_understanding = self.dual_brain.knowledge.load_environment_understanding(self.environment_id)
        
        # 🚀 STEP 2: Integrate new environment context permanently (if provided)
        if env_context:
            print(f"🌍 Integrating environment context for {self.environment_id}...")
            integration_success = self.dual_brain.knowledge.integrate_environment_profile(
                self.environment_id, env_context
            )
            
            if integration_success:
                # Reload understanding after integration
                existing_understanding = self.dual_brain.knowledge.load_environment_understanding(self.environment_id)
            else:
                print(f"⚠️ Failed to integrate environment context")
        
        # 🎯 STEP 3: Agent reasoning about environment knowledge
        self._analyze_environment_understanding(existing_understanding, bool(env_context))
        
        # 🎮 STEP 4: Start enhanced dual brain session with full context
        self.app_context = self.dual_brain.start_enhanced_session(
            self.environment_id,
            env_context=env_context,
            existing_knowledge=existing_understanding  # Pass persistent knowledge
        )

        # 📊 STEP 5: Initialize neural pattern tracking for this match
        self._initialize_match_pattern_tracking()

        # 🚀 STEP 6: Start enhanced logging
        self.logger.start_match(match_id)

        print(f"🆕 Enhanced {self.environment_id} match started with full environment knowledge")
        if self.app_context:
            strategies = len(self.app_context.get('strategies', []))
            lessons = len(self.app_context.get('lessons', []))
            transferable = len(self.app_context.get('transferable_skills', []))
            print(f"   📚 Available knowledge: {strategies} strategies, {lessons} lessons, {transferable} transferable skills")
            
            # Show environment understanding
            understanding_summary = self.dual_brain.knowledge.get_environment_knowledge_summary(self.environment_id)
            print(f"   🧠 Environment knowledge: {understanding_summary}")

    def _analyze_environment_understanding(self, understanding: Dict[str, Any], new_context_provided: bool):
        """
        🧠 Agent reasoning about its environment understanding
        
        This provides insight into what the agent knows/learns about each environment
        """
        if not understanding:
            print(f"🆕 First time encountering {self.environment_id} - starting fresh")
            return
        
        level = understanding.get('understanding_level', 'basic')
        sessions = understanding.get('total_experience', 0)
        knows_objectives = understanding.get('knows_objectives', False)
        knows_rules = understanding.get('knows_rules', False)
        has_framework = understanding.get('has_strategic_framework', False)
        
        # Agent reasoning messages based on understanding level
        if level == 'expert':
            print(f"🏆 Expert level understanding of {understanding.get('display_name', self.environment_id)}")
            print(f"    📊 {sessions} sessions of experience")
            if knows_objectives:
                objective = understanding.get('objectives', {}).get('primary', '')
                print(f"    🎯 I know my objective: {objective}")
            if has_framework:
                skills = understanding.get('strategic_framework', {}).get('core_skills_required', [])
                skill_names = [str(skill) for skill in skills[:3]]
                print(f"    🧠 I've mastered core skills: {', '.join(skill_names)}")
                
        elif level == 'experienced':
            print(f"🧠 Experienced with {understanding.get('display_name', self.environment_id)}")
            print(f"    📊 {sessions} sessions of learning")
            if knows_objectives:
                objective = understanding.get('objectives', {}).get('primary', '')
                print(f"    🎯 I understand the objective: {objective}")
            if knows_rules:
                rules = understanding.get('rules', {}).get('core_mechanics', [])
                print(f"    📋 I know the rules: {len(rules)} core mechanics")
                
        elif level == 'intermediate':
            print(f"🎓 Developing understanding of {understanding.get('display_name', self.environment_id)}")
            print(f"    📊 {sessions} sessions so far")
            if knows_objectives:
                print(f"    🎯 I'm learning the objectives")
            if knows_rules:
                print(f"    📋 I'm learning the rules")
                
        else:  # basic
            print(f"🆕 Basic understanding of {understanding.get('display_name', self.environment_id)}")
            print(f"    📊 {sessions} sessions of experience")
            if new_context_provided:
                print(f"    🌍 Integrating new environment context...")
        
        # Show transferable skills if available
        transferable_skills = understanding.get('transferable_skills', [])
        if transferable_skills:
            skill_names = [str(skill) for skill in transferable_skills[:3]]
            print(f"    🔄 Transferable skills available: {', '.join(skill_names)}")

    def _initialize_match_pattern_tracking(self):
        """Initialize pattern tracking for the current match"""
        # Reset pattern tracking arrays for new match
        self.neural_symbolic_decisions = []
        self.pattern_evolution_history = []

        print("📊 Pattern tracking initialized for match")

    def get_enhanced_environment_awareness(self) -> Dict[str, Any]:
        """
        🧠 Get comprehensive report of agent's environment awareness
        
        This shows what the agent has learned about its current environment
        """
        if not hasattr(self, 'dual_brain'):
            return {}
        
        understanding = self.dual_brain.knowledge.load_environment_understanding(self.environment_id)
        
        awareness = {
            'current_environment': self.environment_id,
            'display_name': understanding.get('display_name', self.environment_id),
            'understanding_level': understanding.get('understanding_level', 'basic'),
            'total_experience': understanding.get('total_experience', 0),
            'environment_type': understanding.get('environment_type', 'unknown'),
            
            # Knowledge categories
            'objectives_known': understanding.get('knows_objectives', False),
            'rules_known': understanding.get('knows_rules', False),
            'strategic_framework_available': understanding.get('has_strategic_framework', False),
            
            # Detailed knowledge
            'primary_objective': understanding.get('objectives', {}).get('primary', ''),
            'core_rules_count': len(understanding.get('rules', {}).get('core_mechanics', [])),
            'core_skills_count': len(understanding.get('strategic_framework', {}).get('core_skills_required', [])),
            'transferable_skills_count': len(understanding.get('transferable_skills', [])),
            
            # Agent learning
            'strategies_learned': len(understanding.get('strategies', [])),
            'lessons_learned': len(understanding.get('lessons', [])),
            'neural_insights_gained': len(understanding.get('neural_insights', [])),
            
            # Experience timeline
            'first_encountered': understanding.get('first_encountered'),
            'last_updated': understanding.get('last_updated')
        }
        
        return awareness

    def get_action(self, raw_state: np.ndarray) -> int:
        """
        Enhanced action selection with Decision Agent for expert selection.
        
        This method:
        1. Gets DDQN action from neural network
        2. Gets Rule-based action from symbolic system
        3. Uses Decision Agent to choose which expert to use
        4. Returns the action from the chosen expert
        """
        try:
            # Store state for later logging
            self.last_state = raw_state.copy()
            
            # STEP 1: Get DDQN action (neural network)
            q_values = self.network.forward(raw_state)
            
            # Epsilon-greedy action selection for DDQN
            if random.random() < self.exploration_rate:
                ddqn_action = random.randint(0, self.action_size - 1)
            else:
                ddqn_action = int(np.argmax(q_values))
            
            self.network.record_decision_pattern(raw_state, ddqn_action, 0.0)  # Reward updated later
            
            # STEP 2: Get Rule-based action (symbolic system)
            rule_action = ddqn_action  # Default to DDQN action if rule system not available
            
            if self.app_context and hasattr(self.dual_brain, 'symbolic_decision_maker'):
                try:
                    # Create action history and reward history for symbolic analysis
                    action_history = [{'action': ddqn_action, 'timestamp': time.time()}]
                    reward_history = [0.0]  # Placeholder

                    rule_action, reasoning, decision_info = self.dual_brain.symbolic_decision_maker.make_enhanced_decision(
                        raw_state, q_values, action_history, reward_history, self.exploration_rate
                    )

                    # Record neural-symbolic decision for compatibility
                    decision_record = {
                        'timestamp': time.time(),
                        'neural_action': ddqn_action,
                        'symbolic_action': rule_action,
                        'decision_info': decision_info,
                        'reasoning': reasoning
                    }
                    self.neural_symbolic_decisions.append(decision_record)

                    # Log the decision
                    if self.logger:
                        self.logger.log_neural_symbolic_decision(decision_info)

                except Exception as e:
                    print(f"⚠️ Symbolic decision error: {e}")
                    # Fall back to DDQN action
                    rule_action = ddqn_action
            else:
                # No symbolic system available, use DDQN action
                rule_action = ddqn_action
            
            # Store actions for logging
            self.last_ddqn_action = ddqn_action
            self.last_rule_action = rule_action
            
            # STEP 3: Use Decision Agent to choose which expert to use
            # Normalize state for Decision Agent (same normalization as network)
            normalized_state = self.network.normalize_input(raw_state)
            expert_choice, final_action = self.decision_agent.choose_expert(
                normalized_state, ddqn_action, rule_action
            )
            
            # Store expert choice for logging
            self.last_expert_choice = expert_choice
            
            self.actions_taken += 1
            return final_action

        except Exception as e:
            print(f"⚠️ Enhanced action error: {e}")
            return random.randint(0, self.action_size - 1)

    def learn(self, reward: float, next_raw_state: np.ndarray, done: bool = False):
        """
        Enhanced learning with neural-symbolic pattern tracking and Decision Agent training.
        
        This method:
        1. Updates DDQN learning (existing)
        2. Logs transition for Decision Agent
        3. Trains Decision Agent
        """
        if self.actions_taken == 0:
            return

        try:
            # Update reward in the last decision pattern
            if self.network.decision_patterns:
                self.network.decision_patterns[-1]['reward'] = reward

            # Track pattern evolution
            enhanced_features = self.network.get_enhanced_core_features()
            pattern_info = {
                'timestamp': time.time(),
                'pattern_stability': enhanced_features['pattern_stability'],
                'feature_summary': enhanced_features['feature_summary'],
                'reward_correlations': enhanced_features['reward_correlations'],
                'transfer_readiness': enhanced_features['pattern_stability']  # Simplified metric
            }
            self.pattern_evolution_history.append(pattern_info)

            # Log pattern evolution
            if self.logger:
                self.logger.log_neural_pattern_evolution(pattern_info)

            # Update neural-symbolic decision correlation if applicable
            if self.neural_symbolic_decisions:
                last_decision = self.neural_symbolic_decisions[-1]
                if 'decision_info' in last_decision:
                    decision_info = last_decision['decision_info']
                    if hasattr(self.dual_brain, 'symbolic_decision_maker'):
                        self.dual_brain.symbolic_decision_maker.update_neural_symbolic_correlation(decision_info,
                                                                                                   reward)

            # Standard learning process
            self.match_reward += reward
            self.total_reward += reward

            # Update exploration
            if self.training_steps % 100 == 0:
                if self.exploration_rate > self.min_exploration:
                    self.exploration_rate *= self.exploration_decay

            self.training_steps += 1

            # NEW: Log transition for Decision Agent and train it
            if (self.last_state is not None and 
                self.last_ddqn_action is not None and 
                self.last_rule_action is not None and 
                self.last_expert_choice is not None):
                
                # Normalize states for Decision Agent
                normalized_last_state = self.network.normalize_input(self.last_state)
                normalized_next_state = self.network.normalize_input(next_raw_state) if next_raw_state is not None else None
                
                # Record transition for Decision Agent
                self.decision_agent.record_transition(
                    state=normalized_last_state,
                    ddqn_action=self.last_ddqn_action,
                    rule_action=self.last_rule_action,
                    expert_choice=self.last_expert_choice,
                    reward=reward,
                    next_state=normalized_next_state,
                    done=done
                )
                
                # Train Decision Agent
                self.decision_agent.train()

                # Record symbolic experiment outcome for knowledge corpus
                if (self.dual_brain and hasattr(self.dual_brain, 'knowledge')
                        and self.neural_symbolic_decisions):
                    last_decision = self.neural_symbolic_decisions[-1]
                    decision_info = last_decision.get('decision_info', {})
                    experiment_payload = {
                        'timestamp': time.time(),
                        'environment': self.environment_id,
                        'strategy': decision_info.get('strategy') or decision_info.get('skill_id') or decision_info.get(
                            'decision_type'),
                        'decision_type': decision_info.get('decision_type'),
                        'reward': reward,
                        'success': reward > 0,
                        'expert_choice': self.last_expert_choice,
                        'notes': last_decision.get('reasoning')
                    }
                    try:
                        self.dual_brain.knowledge.record_experiment_outcome(experiment_payload)
                    except Exception as experiment_error:
                        print(f"⚠️ Unable to log experiment outcome: {experiment_error}")

            # Extract transferable insights
            self._extract_enhanced_transferable_insights(reward, done, enhanced_features)

        except Exception as e:
            print(f"❌ Enhanced learn error: {e}")

    def _extract_enhanced_transferable_insights(self, reward: float, done: bool, enhanced_features: Dict):
        """Extract enhanced insights that could transfer to other environments"""
        try:
            # High-level insights based on performance and patterns
            if reward > 5.0 and enhanced_features['pattern_stability'] > 0.7:
                insight = {
                    'type': 'high_performance_stable_pattern',
                    'reward': reward,
                    'pattern_stability': enhanced_features['pattern_stability'],
                    'core_features_snapshot': enhanced_features['feature_summary'],
                    'environment': self.environment_id,
                    'timestamp': time.time(),
                    'transferable_principle': 'Stable neural patterns correlate with high performance'
                }
                self.cross_environment_insights.append(insight)

            elif reward < -3.0:
                insight = {
                    'type': 'failure_pattern_with_instability',
                    'reward': reward,
                    'pattern_stability': enhanced_features['pattern_stability'],
                    'environment': self.environment_id,
                    'timestamp': time.time(),
                    'transferable_principle': 'Unstable patterns lead to poor performance'
                }
                self.cross_environment_insights.append(insight)

            # Keep only recent insights
            if len(self.cross_environment_insights) > 50:
                self.cross_environment_insights = self.cross_environment_insights[-50:]

        except Exception as e:
            print(f"❌ Error extracting enhanced transferable insights: {e}")

    def end_match(self, winner: str, final_scores: Dict, game_stats: Dict):
        """End match with enhanced neural-symbolic analysis"""
        try:
            # Calculate enhanced transferable performance metrics
            enhanced_metrics = self._calculate_enhanced_transferable_metrics()

            # Update enhanced agent capabilities
            self._update_enhanced_agent_capabilities(enhanced_metrics)

            # End enhanced dual brain session
            if hasattr(self.dual_brain, 'end_enhanced_session'):
                self.dual_brain.end_enhanced_session(winner == "Agent Byte", self.match_reward,
                                                     "Enhanced match completed")
            else:
                # Fallback to regular end session
                self.dual_brain.end_session(winner == "Agent Byte", self.match_reward, "Enhanced match completed")

            # End enhanced logging with neural-symbolic insights
            enhanced_stats = {**game_stats, **enhanced_metrics}
            self.logger.end_match(winner, final_scores, enhanced_stats)

            # Update knowledge progression/unlocks for this environment
            if hasattr(self.dual_brain, 'knowledge'):
                progression_payload = {
                    'win': winner == "Agent Byte",
                    'reward': getattr(self, 'match_reward', 0),
                    'pattern_stability': enhanced_metrics.get('pattern_stability_score', 0.0),
                    'symbolic_coherence': enhanced_metrics.get('neural_symbolic_coherence', 0.0),
                    'skills_effectiveness': enhanced_metrics.get('transferable_skills_effectiveness', 0.0)
                }
                try:
                    self.dual_brain.knowledge.update_environment_progression(progression_payload)
                except Exception as progression_error:
                    print(f"⚠️ Unable to update knowledge progression: {progression_error}")

            # Save enhanced progress
            self._save_enhanced_all_progress()

            self.games_played += 1
            if winner == "Agent Byte":
                self.wins += 1

            print(f"🏁 Enhanced {self.environment_id} match ended: {winner} wins!")
            print(f"   🔄 Transferable skills used: {len(self.transferable_skills_used)}")
            print(f"   📊 Transfer events: {len(self.knowledge_transfer_events)}")
            print(f"   🧠🧩 Neural-symbolic decisions: {len(self.neural_symbolic_decisions)}")
            print(f"   📊 Pattern evolution events: {len(self.pattern_evolution_history)}")
            print(f"   🌍 Cross-env insights: {len(self.cross_environment_insights)}")

            return enhanced_stats

        except Exception as e:
            print(f"❌ Error in enhanced end_match: {e}")
            return game_stats

    def _calculate_enhanced_transferable_metrics(self) -> Dict[str, Any]:
        """Calculate enhanced metrics relevant to neural-symbolic transfer learning"""
        try:
            # Enhanced transferable skills effectiveness with pattern analysis
            skills_effectiveness = []
            pattern_stability_scores = []

            for skill in self.transferable_skills_used:
                effectiveness = skill.get('effectiveness', 0.5)
                stability = skill.get('pattern_stability', 0.5)
                skills_effectiveness.append(effectiveness)
                pattern_stability_scores.append(stability)

            avg_skills_effectiveness = np.mean(skills_effectiveness) if skills_effectiveness else 0.0
            avg_pattern_stability = np.mean(pattern_stability_scores) if pattern_stability_scores else 0.0

            # Neural-symbolic decision coherence
            neural_decisions = len([d for d in self.neural_symbolic_decisions
                                    if d.get('decision_info', {}).get('decision_type') == 'neural'])
            symbolic_decisions = len([d for d in self.neural_symbolic_decisions
                                      if d.get('decision_info', {}).get('decision_type') == 'symbolic'])
            total_decisions = max(1, neural_decisions + symbolic_decisions)

            decision_balance = symbolic_decisions / total_decisions

            # Pattern evolution analysis
            if len(self.pattern_evolution_history) >= 2:
                early_stability = np.mean([p['pattern_stability'] for p in
                                           self.pattern_evolution_history[:len(self.pattern_evolution_history) // 2]])
                late_stability = np.mean([p['pattern_stability'] for p in
                                          self.pattern_evolution_history[len(self.pattern_evolution_history) // 2:]])
                pattern_improvement = late_stability - early_stability
            else:
                pattern_improvement = 0.0

            # Knowledge transfer success rate
            successful_transfers = len([event for event in self.knowledge_transfer_events
                                        if event.get('success_rate', 0) > 0.5])
            transfer_success_rate = successful_transfers / max(1, len(self.knowledge_transfer_events))

            return {
                'transferable_skills_effectiveness': avg_skills_effectiveness,
                'pattern_stability_score': avg_pattern_stability,
                'neural_symbolic_coherence': decision_balance,
                'pattern_improvement_trajectory': pattern_improvement,
                'knowledge_transfer_success_rate': transfer_success_rate,
                'cross_environment_insights_generated': len(self.cross_environment_insights),
                'adaptation_speed': self.training_steps / max(1, self.games_played),
                'enhanced_pattern_recognition': avg_pattern_stability * avg_skills_effectiveness,
                'transfer_readiness_score': min(1.0,
                                                avg_skills_effectiveness * avg_pattern_stability * decision_balance)
            }

        except Exception as e:
            print(f"❌ Error calculating enhanced transferable metrics: {e}")
            return {}

    def _update_enhanced_agent_capabilities(self, enhanced_metrics: Dict[str, Any]):
        """Update enhanced agent capabilities in profile"""
        try:
            capabilities = self.profile.get('enhanced_capabilities', {})
            neural_symbolic = self.profile.get('neural_symbolic_integration', {})

            # Update enhanced transferable skills
            if self.transferable_skills_used:
                for skill in self.transferable_skills_used:
                    skill_record = {
                        'name': skill['name'],
                        'environment': self.environment_id,
                        'confidence': skill.get('effectiveness', 0.5),
                        'pattern_stability': skill.get('pattern_stability', 0.5),
                        'neural_symbolic_derived': True,
                        'last_used': time.time()
                    }

                    # Add or update skill
                    existing_skills = capabilities.get('transferable_skills', [])
                    skill_found = False
                    for existing_skill in existing_skills:
                        if existing_skill['name'] == skill['name']:
                            existing_skill.update(skill_record)
                            skill_found = True
                            break

                    if not skill_found:
                        existing_skills.append(skill_record)

                    capabilities['transferable_skills'] = existing_skills

            # Update enhanced capabilities
            capabilities['learning_efficiency'] = enhanced_metrics.get('transferable_skills_effectiveness', 0.0)
            capabilities['adaptation_speed'] = enhanced_metrics.get('adaptation_speed', 0.0)
            capabilities['neural_symbolic_coherence'] = enhanced_metrics.get('neural_symbolic_coherence', 0.0)
            capabilities['pattern_recognition_ability'] = enhanced_metrics.get('enhanced_pattern_recognition', 0.0)
            capabilities['cross_environment_transfer_rate'] = enhanced_metrics.get('knowledge_transfer_success_rate',
                                                                                   0.0)

            # Update neural-symbolic integration metrics
            neural_symbolic['decision_correlation_rate'] = enhanced_metrics.get('neural_symbolic_coherence', 0.0)
            neural_symbolic['pattern_stability_score'] = enhanced_metrics.get('pattern_stability_score', 0.0)
            neural_symbolic['transfer_effectiveness'] = enhanced_metrics.get('transfer_readiness_score', 0.0)

            self.profile['enhanced_capabilities'] = capabilities
            self.profile['neural_symbolic_integration'] = neural_symbolic
            self._save_enhanced_agent_profile()

        except Exception as e:
            print(f"❌ Error updating enhanced agent capabilities: {e}")

    def _save_enhanced_all_progress(self):
        """Save all enhanced agent progress including networks, neural patterns, knowledge, and Decision Agent"""
        try:
            # Save enhanced neural networks with pattern data
            self.network.save_network(self.network_file)

            # Save enhanced dual brain system
            if hasattr(self.dual_brain, 'save_enhanced_all'):
                self.dual_brain.save_enhanced_all()
            else:
                # Fallback to regular save
                self.dual_brain.save_all()

            # Save neural pattern evolution data
            self._save_neural_pattern_data()

            # Save decision correlation data
            self._save_decision_correlation_data()
            
            # Save Decision Agent
            self.decision_agent.save(self.decision_agent_file)

            # Update profile with current environment if not already present
            if self.environment_id not in self.profile.get('environments', []):
                self.profile['environments'].append(self.environment_id)
                self._save_enhanced_agent_profile()

            return True

        except Exception as e:
            print(f"❌ Error saving enhanced progress: {e}")
            return False

    def _save_neural_pattern_data(self):
        """Save neural pattern evolution data"""
        try:
            pattern_data = {
                'agent_id': self.agent_id,
                'environment_id': self.environment_id,
                'saved_at': datetime.datetime.now().isoformat(),
                'architecture_version': 'Agent Byte v2.1 - Neural-Symbolic Integration',
                'pattern_evolution_history': self.pattern_evolution_history[-100:],  # Keep recent
                'transferable_insights': self.cross_environment_insights[-50:],
                'pattern_statistics': {
                    'total_patterns_tracked': len(self.pattern_evolution_history),
                    'average_stability': np.mean([p['pattern_stability'] for p in
                                                  self.pattern_evolution_history]) if self.pattern_evolution_history else 0.0,
                    'stability_improvement': self._calculate_pattern_improvement(),
                    'transfer_readiness': self._calculate_pattern_transfer_readiness()
                }
            }

            with open(self.neural_patterns_file, 'w') as f:
                json.dump(pattern_data, f, indent=2)

            print(f"🔍 Neural pattern data saved: {len(self.pattern_evolution_history)} patterns tracked")

        except Exception as e:
            print(f"❌ Error saving neural pattern data: {e}")

    def _save_decision_correlation_data(self):
        """Save neural-symbolic decision correlation data"""
        try:
            correlation_data = {
                'agent_id': self.agent_id,
                'environment_id': self.environment_id,
                'saved_at': datetime.datetime.now().isoformat(),
                'neural_symbolic_decisions': self.neural_symbolic_decisions[-100:],  # Keep recent
                'correlation_statistics': {
                    'total_decisions': len(self.neural_symbolic_decisions),
                    'neural_decisions': len([d for d in self.neural_symbolic_decisions
                                             if d.get('decision_info', {}).get('decision_type') == 'neural']),
                    'symbolic_decisions': len([d for d in self.neural_symbolic_decisions
                                               if d.get('decision_info', {}).get('decision_type') == 'symbolic']),
                    'average_coherence': self._calculate_decision_coherence(),
                    'integration_effectiveness': self._calculate_integration_effectiveness()
                }
            }

            with open(self.decision_correlations_file, 'w') as f:
                json.dump(correlation_data, f, indent=2)

            print(f"🔗 Decision correlation data saved: {len(self.neural_symbolic_decisions)} decisions tracked")

        except Exception as e:
            print(f"❌ Error saving decision correlation data: {e}")

    def _calculate_pattern_improvement(self) -> float:
        """Calculate pattern stability improvement over time"""
        if len(self.pattern_evolution_history) < 10:
            return 0.0

        early_patterns = self.pattern_evolution_history[:len(self.pattern_evolution_history) // 2]
        late_patterns = self.pattern_evolution_history[len(self.pattern_evolution_history) // 2:]

        early_stability = np.mean([p['pattern_stability'] for p in early_patterns])
        late_stability = np.mean([p['pattern_stability'] for p in late_patterns])

        return late_stability - early_stability

    def _calculate_pattern_transfer_readiness(self) -> float:
        """Calculate how ready patterns are for transfer to other environments"""
        if not self.pattern_evolution_history:
            return 0.0

        recent_patterns = self.pattern_evolution_history[-20:] if len(
            self.pattern_evolution_history) >= 20 else self.pattern_evolution_history
        stability_scores = [p['pattern_stability'] for p in recent_patterns]

        avg_stability = np.mean(stability_scores)
        stability_consistency = 1.0 - np.var(stability_scores) if len(stability_scores) > 1 else 0.0

        return (avg_stability * 0.7 + stability_consistency * 0.3)

    def _calculate_decision_coherence(self) -> float:
        """Calculate coherence between neural and symbolic decisions"""
        if not self.neural_symbolic_decisions:
            return 0.0

        coherent_decisions = 0
        total_decisions = len(self.neural_symbolic_decisions)

        for decision in self.neural_symbolic_decisions:
            decision_info = decision.get('decision_info', {})
            neural_action = decision_info.get('neural_action')
            symbolic_action = decision_info.get('symbolic_action')

            if neural_action == symbolic_action:
                coherent_decisions += 1

        return coherent_decisions / total_decisions

    def _calculate_integration_effectiveness(self) -> float:
        """Calculate effectiveness of neural-symbolic integration"""
        if not self.neural_symbolic_decisions:
            return 0.0

        symbolic_decisions = [d for d in self.neural_symbolic_decisions
                              if d.get('decision_info', {}).get('decision_type') == 'symbolic']

        if not symbolic_decisions:
            return 0.0

        # Integration effectiveness based on symbolic decision success
        integration_rate = len(symbolic_decisions) / len(self.neural_symbolic_decisions)
        coherence_rate = self._calculate_decision_coherence()

        return (integration_rate * 0.6 + coherence_rate * 0.4)

    def get_enhanced_transfer_readiness_report(self) -> Dict[str, Any]:
        """Generate enhanced report on agent's readiness for transfer learning"""
        try:
            capabilities = self.profile.get('enhanced_capabilities', {})
            neural_symbolic = self.profile.get('neural_symbolic_integration', {})
            transferable_skills = capabilities.get('transferable_skills', [])

            # Enhanced readiness calculation
            pattern_readiness = self._calculate_pattern_transfer_readiness()
            decision_coherence = neural_symbolic.get('decision_correlation_rate', 0.0)
            skills_confidence = np.mean(
                [s.get('confidence', 0) for s in transferable_skills]) if transferable_skills else 0.0

            transfer_readiness_score = (pattern_readiness * 0.4 + decision_coherence * 0.3 + skills_confidence * 0.3)

            return {
                'agent_id': self.agent_id,
                'current_environment': self.environment_id,
                'architecture_version': 'Agent Byte v2.1 - Neural-Symbolic Integration',
                'total_environments_experienced': len(self.profile.get('environments', [])),
                'transferable_skills_learned': len(transferable_skills),
                'enhanced_capabilities': capabilities,
                'neural_symbolic_integration': neural_symbolic,
                'pattern_analysis': {
                    'pattern_stability_score': pattern_readiness,
                    'pattern_improvement': self._calculate_pattern_improvement(),
                    'patterns_tracked': len(self.pattern_evolution_history)
                },
                'decision_analysis': {
                    'neural_symbolic_coherence': decision_coherence,
                    'integration_effectiveness': self._calculate_integration_effectiveness(),
                    'decisions_tracked': len(self.neural_symbolic_decisions)
                },
                'transfer_readiness_score': transfer_readiness_score,
                'recommended_next_environments': self._recommend_enhanced_next_environments(),
                'enhanced_network_architecture': f"256→{self.network.core_sizes}→{self.network.adapter_size}→{self.action_size}",
                'neural_symbolic_compatible': True
            }

        except Exception as e:
            print(f"❌ Error generating enhanced transfer readiness report: {e}")
            return {}

    def _recommend_enhanced_next_environments(self) -> List[str]:
        """Recommend next environments based on enhanced neural-symbolic analysis"""
        capabilities = self.profile.get('enhanced_capabilities', {})
        learned_skills = [skill['name'] for skill in capabilities.get('transferable_skills', [])]

        recommendations = []

        # Enhanced recommendations based on neural-symbolic capabilities
        if any('prediction' in skill.lower() for skill in learned_skills):
            recommendations.extend(['trajectory_prediction_games', 'forecasting_challenges'])
        if any('timing' in skill.lower() for skill in learned_skills):
            recommendations.extend(['rhythm_based_games', 'real_time_strategy'])
        if any('strategy' in skill.lower() or 'positioning' in skill.lower() for skill in learned_skills):
            recommendations.extend(['strategic_games', 'spatial_reasoning_tasks'])
        if capabilities.get('neural_symbolic_coherence', 0) > 0.7:
            recommendations.extend(['complex_reasoning_environments', 'multi_domain_challenges'])
        if capabilities.get('pattern_recognition_ability', 0) > 0.8:
            recommendations.extend(['pattern_matching_games', 'anomaly_detection_tasks'])

        # Remove duplicates and return top recommendations
        return list(dict.fromkeys(recommendations))[:5]

    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced agent statistics with neural-symbolic metrics and Decision Agent stats"""
        capabilities = self.profile.get('enhanced_capabilities', {})
        neural_symbolic = self.profile.get('neural_symbolic_integration', {})
        
        # Get Decision Agent stats
        decision_agent_stats = self.decision_agent.get_stats() if hasattr(self, 'decision_agent') else {}

        return {
            'games_played': self.games_played,
            'wins': self.wins,
            'win_rate': (self.wins / max(1, self.games_played)) * 100,
            'total_reward': self.total_reward,
            'training_steps': self.training_steps,
            'exploration_rate': self.exploration_rate,
            'actions_taken': self.actions_taken,

            # Enhanced neural-symbolic metrics
            'neural_symbolic_integration': neural_symbolic.get('enabled', False),
            'pattern_stability_score': neural_symbolic.get('pattern_stability_score', 0.0),
            'decision_correlation_rate': neural_symbolic.get('decision_correlation_rate', 0.0),
            'transfer_effectiveness': neural_symbolic.get('transfer_effectiveness', 0.0),
            'transferable_skills_count': len(capabilities.get('transferable_skills', [])),
            'cross_environment_ready': len(self.profile.get('environments', [])) > 1,
            'enhanced_learning_efficiency': capabilities.get('learning_efficiency', 0.0),
            'neural_symbolic_coherence': capabilities.get('neural_symbolic_coherence', 0.0),
            'pattern_recognition_ability': capabilities.get('pattern_recognition_ability', 0.0),

            # Decision Agent metrics
            'decision_agent': {
                'ddqn_choices': decision_agent_stats.get('ddqn_choices', 0),
                'rule_choices': decision_agent_stats.get('rule_choices', 0),
                'ddqn_ratio': decision_agent_stats.get('ddqn_ratio', 0.0),
                'rule_ratio': decision_agent_stats.get('rule_ratio', 0.0),
                'exploration_rate': decision_agent_stats.get('exploration_rate', 0.0),
                'training_steps': decision_agent_stats.get('training_steps', 0),
                'replay_buffer_size': decision_agent_stats.get('replay_buffer_size', 0)
            },

            # Current session metrics
            'current_match_reward': self.match_reward,
            'current_session_patterns': len(self.pattern_evolution_history),
            'current_session_decisions': len(self.neural_symbolic_decisions),
            'current_session_insights': len(self.cross_environment_insights)
        }


# Alias for backward compatibility
AgentByte = EnhancedAgentByte

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Enhanced Agent Byte v2.1 with Neural-Symbolic Integration...")

    # Create enhanced test agent
    agent = EnhancedAgentByte(
        agent_id="test_enhanced_agent_001",
        environment_id="pong",
        raw_state_size=14,
        action_size=3
    )

    # Test enhanced network with pattern tracking
    test_state = np.random.random(14)
    q_values = agent.network.forward(test_state)
    enhanced_features = agent.network.get_enhanced_core_features()

    print(f"\n🧠 Enhanced Network Test:")
    print(f"   Input: {test_state.shape} → Output: {q_values.shape}")
    print(f"   Pattern stability: {enhanced_features['pattern_stability']:.2f}")
    print(f"   Feature summary: {enhanced_features['feature_summary']}")

    # Test enhanced transferable skills with neural-symbolic integration
    agent.start_new_match("enhanced_test_match_001")

    # Simulate enhanced learning with neural-symbolic decisions
    for i in range(20):
        action = agent.get_action(test_state)
        reward = random.uniform(-1, 3)
        agent.learn(reward, test_state)

        if i % 5 == 0:
            print(f"Enhanced Step {i + 1}: Action={action}, Reward={reward:.2f}")
            if agent.neural_symbolic_decisions:
                last_decision = agent.neural_symbolic_decisions[-1]
                decision_type = last_decision.get('decision_info', {}).get('decision_type', 'unknown')
                print(f"   Decision type: {decision_type}")

    # End enhanced match and get comprehensive analysis
    enhanced_stats = agent.end_match("Agent Byte", {"agent": 15, "opponent": 12}, {"total_actions": 100})

    # Get enhanced transfer readiness report
    enhanced_readiness_report = agent.get_enhanced_transfer_readiness_report()

    print(f"\n📊 Enhanced Transfer Readiness Report:")
    print(f"   Architecture: {enhanced_readiness_report.get('architecture_version')}")
    print(f"   Transfer Readiness Score: {enhanced_readiness_report.get('transfer_readiness_score', 0):.2f}")
    print(
        f"   Pattern Stability: {enhanced_readiness_report.get('pattern_analysis', {}).get('pattern_stability_score', 0):.2f}")
    print(
        f"   Neural-Symbolic Coherence: {enhanced_readiness_report.get('decision_analysis', {}).get('neural_symbolic_coherence', 0):.2f}")
    print(f"   Transferable Skills: {enhanced_readiness_report.get('transferable_skills_learned', 0)}")
    print(f"   Recommended Environments: {enhanced_readiness_report.get('recommended_next_environments', [])}")

    # Get enhanced stats
    enhanced_stats = agent.get_stats()
    print(f"\n📈 Enhanced Agent Stats:")
    print(f"   Neural-Symbolic Integration: {enhanced_stats.get('neural_symbolic_integration')}")
    print(f"   Pattern Recognition Ability: {enhanced_stats.get('pattern_recognition_ability', 0):.2f}")
    print(f"   Cross-Environment Ready: {enhanced_stats.get('cross_environment_ready')}")

    # Test enhanced saving
    save_success = agent._save_enhanced_all_progress()
    print(f"\n💾 Enhanced Save Test: {'✅ Success' if save_success else '❌ Failed'}")

    print(f"\n✅ Enhanced Agent Byte v2.1 with Neural-Symbolic Integration test complete!")
    print(f"🧠🧩 Neural and symbolic brains working together for true intelligence!")
    print(f"🔗 Pattern tracking enables deep understanding of learning process!")
    print(f"🌍 Ready for universal transfer learning with enhanced coherence!")
    print(f"Neural-symbolic integration: Active")