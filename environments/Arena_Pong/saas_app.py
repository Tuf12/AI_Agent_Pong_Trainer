# saas_app.py - Enhanced SaaS Platform with Multi-Environment Transfer Learning Support
from flask import Flask, request, jsonify, session, render_template
from flask_socketio import SocketIO, emit, join_room
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS
import json
import time
import threading
import uuid
import os
import shutil
from datetime import datetime, timezone
from typing import Dict, Optional, List, Any
from copy import deepcopy

# Import enhanced Agent Byte components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from agent_byte import AgentByte
from adapters.pong_arena_adapter import PongArenaAdapter
from data_pipeline.dataset_builder import build_and_export_dataset
                                    
# Updated to use Arena Pong Environment v2.0 with 256-dimension support
from arena_pong_environment import ArenaPongEnvironment
from pong_decision_logger import PongDecisionLogger
from pong_toolbox import PongToolbox

# Flask setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'agent_byte_saas_v2_transferable_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///agent_byte_saas_v2.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

DATASET_EXPORT_DIR = os.path.join(os.path.dirname(__file__), 'dataset_exports')
os.makedirs(DATASET_EXPORT_DIR, exist_ok=True)

# Global game sessions
active_games: Dict[str, 'GameSession'] = {}


# Enhanced Database Models with Multi-Environment Support
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    subscription_tier = db.Column(db.String(20), default='free')

    # Enhanced user metrics
    total_agents_created = db.Column(db.Integer, default=0)
    total_environments_explored = db.Column(db.Integer, default=0)
    total_training_time = db.Column(db.Float, default=0.0)

    agents = db.relationship('Agent', backref='owner', lazy=True, cascade='all, delete-orphan')
    matches = db.relationship('Match', foreign_keys='Match.user_id', backref='user', lazy=True)


    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'subscription_tier': self.subscription_tier,
            'agent_count': len(self.agents),
            'total_agents_created': self.total_agents_created,
            'total_environments_explored': self.total_environments_explored,
            'total_training_time': round(self.total_training_time, 1)
        }


class Agent(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    last_trained = db.Column(db.DateTime)

    # Multi-environment support
    environments_experienced = db.Column(db.Text, default='[]')  # JSON list of environment names
    primary_environment = db.Column(db.String(50), default='pong')

    # Enhanced agent metrics
    total_wins = db.Column(db.Integer, default=0)
    total_losses = db.Column(db.Integer, default=0)
    total_training_time = db.Column(db.Float, default=0.0)
    elo_rating = db.Column(db.Integer, default=1200)

    # Transfer learning metrics
    transferable_skills_count = db.Column(db.Integer, default=0)
    knowledge_transfer_success_rate = db.Column(db.Float, default=0.0)
    cross_environment_performance = db.Column(db.Text, default='{}')  # JSON dict
    transfer_learning_maturity = db.Column(db.Float, default=0.0)

    # Architecture metadata
    architecture_version = db.Column(db.String(50), default='Agent Byte v2.0 - Transferable')
    file_structure_version = db.Column(db.String(10), default='2.0')

    def get_environments_experienced(self) -> List[str]:
        """Get list of environments this agent has experienced"""
        try:
            return json.loads(self.environments_experienced)
        except:
            return [self.primary_environment] if self.primary_environment else ['pong']

    def add_environment_experience(self, environment_name: str):
        """Add new environment to agent's experience"""
        try:
            envs = self.get_environments_experienced()
            if environment_name not in envs:
                envs.append(environment_name)
                self.environments_experienced = json.dumps(envs)
                return True
            return False
        except:
            self.environments_experienced = json.dumps([environment_name])
            return True

    def get_cross_environment_performance(self) -> Dict[str, Any]:
        """Get cross-environment performance data"""
        try:
            return json.loads(self.cross_environment_performance)
        except:
            return {}

    def update_cross_environment_performance(self, environment: str, performance_data: Dict[str, Any]):
        """Update performance data for specific environment"""
        try:
            perf_data = self.get_cross_environment_performance()
            perf_data[environment] = performance_data
            self.cross_environment_performance = json.dumps(perf_data)
        except Exception as e:
            print(f"Error updating cross-environment performance: {e}")

    def to_dict(self):
        total_games = self.total_wins + self.total_losses
        win_rate = (self.total_wins / max(1, total_games)) * 100
        environments = self.get_environments_experienced()

        return {
            'id': self.id,
            'name': self.name,
            'primary_environment': self.primary_environment,
            'environments_experienced': environments,
            'created_at': self.created_at.isoformat(),
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'stats': {
                'wins': self.total_wins,
                'losses': self.total_losses,
                'win_rate': round(win_rate, 1),
                'elo_rating': self.elo_rating,
                'training_time': round(self.total_training_time, 1),
                'environments_count': len(environments),
                'transferable_skills': self.transferable_skills_count,
                'transfer_success_rate': round(self.knowledge_transfer_success_rate * 100, 1),
                'transfer_maturity': round(self.transfer_learning_maturity, 2)
            },
            'transfer_learning': {
                'maturity_score': self.transfer_learning_maturity,
                'skills_count': self.transferable_skills_count,
                'success_rate': self.knowledge_transfer_success_rate,
                'cross_environment_ready': len(environments) > 1
            },
            'architecture_version': self.architecture_version
        }


class Match(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    agent1_id = db.Column(db.String(36), db.ForeignKey('agent.id'), nullable=False)
    agent2_id = db.Column(db.String(36), db.ForeignKey('agent.id'), nullable=True)
    environment = db.Column(db.String(50), default='pong')
    status = db.Column(db.String(20), default='waiting')
    winner = db.Column(db.String(20))
    final_score = db.Column(db.Text)
    started_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    completed_at = db.Column(db.DateTime)

    # Enhanced match data
    match_type = db.Column(db.String(20), default='user_vs_agent')
    spectator_count = db.Column(db.Integer, default=0)
    tournament_id = db.Column(db.String(36), nullable=True)
    is_public = db.Column(db.Boolean, default=True)

    # Transfer learning metrics for this match
    transfer_events_count = db.Column(db.Integer, default=0)
    transferable_skills_used = db.Column(db.Integer, default=0)
    knowledge_transfer_effectiveness = db.Column(db.Float, default=0.0)

    agent1 = db.relationship('Agent', foreign_keys=[agent1_id], backref='matches_as_agent1')
    agent2 = db.relationship('Agent', foreign_keys=[agent2_id], backref='matches_as_agent2')

    def to_dict(self):
        return {
            'id': self.id,
            'agent1_id': self.agent1_id,
            'agent2_id': self.agent2_id,
            'environment': self.environment,
            'status': self.status,
            'winner': self.winner,
            'final_score': json.loads(self.final_score) if self.final_score else None,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'agent1_name': self.agent1.name if self.agent1 else 'Unknown',
            'agent2_name': self.agent2.name if self.agent2 else 'User',
            'match_type': self.match_type,
            'spectator_count': self.spectator_count,
            'is_public': self.is_public,
            'transfer_learning': {
                'events_count': self.transfer_events_count,
                'skills_used': self.transferable_skills_used,
                'effectiveness': round(self.knowledge_transfer_effectiveness, 3)
            }
        }


class UserPreferenceSnapshot(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    preferences = db.Column(db.Text, default='{}')  # JSON blob describing control scheme, ux prefs, etc.
    learning_goals = db.Column(db.Text, default='[]')  # JSON list of goal objects
    agent_focus = db.Column(db.Text, default='[]')  # JSON list of agent identifiers/styles
    notes = db.Column(db.Text)

    user = db.relationship('User', backref=db.backref('preference_snapshots', lazy=True, cascade='all, delete-orphan'))

    def to_dict(self):
        def _safe_load(payload, default):
            try:
                return json.loads(payload) if payload else default
            except Exception:
                return default

        return {
            'id': self.id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'preferences': _safe_load(self.preferences, {}),
            'learning_goals': _safe_load(self.learning_goals, []),
            'agent_focus': _safe_load(self.agent_focus, []),
            'notes': self.notes or ''
        }


class AgentConfigSnapshot(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = db.Column(db.String(36), db.ForeignKey('agent.id'), nullable=False)
    match_id = db.Column(db.String(36), db.ForeignKey('match.id'))
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    config_data = db.Column(db.Text, default='{}')
    decision_agent_config = db.Column(db.Text, default='{}')

    agent = db.relationship('Agent', backref=db.backref('config_snapshots', lazy=True, cascade='all, delete-orphan'))

    def to_dict(self):
        def _safe_load(payload):
            try:
                return json.loads(payload) if payload else {}
            except Exception:
                return {}

        return {
            'id': self.id,
            'agent_id': self.agent_id,
            'match_id': self.match_id,
            'created_at': self.created_at.isoformat(),
            'config_data': _safe_load(self.config_data),
            'decision_agent_config': _safe_load(self.decision_agent_config)
        }


class TrajectoryRun(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    match_id = db.Column(db.String(36), db.ForeignKey('match.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    agent_id = db.Column(db.String(36), db.ForeignKey('agent.id'), nullable=False)
    environment = db.Column(db.String(50), default='pong')
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    completed_at = db.Column(db.DateTime)
    total_steps = db.Column(db.Integer, default=0)
    total_reward = db.Column(db.Float, default=0.0)
    run_metadata = db.Column('metadata', db.Text, default='{}')
    agent_config_snapshot_id = db.Column(db.String(36), db.ForeignKey('agent_config_snapshot.id'))
    user_pref_snapshot_id = db.Column(db.String(36), db.ForeignKey('user_preference_snapshot.id'))

    steps = db.relationship('TrajectoryStep', backref='run', lazy=True, cascade='all, delete-orphan')

    def to_dict(self, include_steps: bool = False):
        try:
            metadata = json.loads(self.run_metadata) if self.run_metadata else {}
        except Exception:
            metadata = {}

        data = {
            'id': self.id,
            'match_id': self.match_id,
            'user_id': self.user_id,
            'agent_id': self.agent_id,
            'environment': self.environment,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'total_steps': self.total_steps,
            'total_reward': self.total_reward,
            'metadata': metadata,
            'agent_config_snapshot_id': self.agent_config_snapshot_id,
            'user_pref_snapshot_id': self.user_pref_snapshot_id
        }

        if include_steps:
            data['steps'] = [step.to_dict() for step in self.steps]
        return data


class TrajectoryStep(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.String(36), db.ForeignKey('trajectory_run.id'), nullable=False)
    step_index = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    state_vector = db.Column(db.Text, nullable=False)  # JSON encoded list
    next_state_vector = db.Column(db.Text)
    normalized_state_vector = db.Column(db.Text)
    normalized_next_state_vector = db.Column(db.Text)
    action = db.Column(db.Integer)
    reward = db.Column(db.Float)
    done = db.Column(db.Boolean, default=False)
    expert_choice = db.Column(db.Integer)
    rule_action = db.Column(db.Integer)
    ddqn_action = db.Column(db.Integer)
    step_metadata = db.Column('metadata', db.Text, default='{}')

    def to_dict(self):
        def _safe_load(payload):
            try:
                return json.loads(payload) if payload else None
            except Exception:
                return None

        return {
            'id': self.id,
            'run_id': self.run_id,
            'step_index': self.step_index,
            'timestamp': self.timestamp.isoformat(),
            'state_vector': _safe_load(self.state_vector),
            'next_state_vector': _safe_load(self.next_state_vector),
            'normalized_state_vector': _safe_load(self.normalized_state_vector),
            'normalized_next_state_vector': _safe_load(self.normalized_next_state_vector),
            'action': self.action,
            'reward': self.reward,
            'done': self.done,
            'expert_choice': self.expert_choice,
            'rule_action': self.rule_action,
            'ddqn_action': self.ddqn_action,
            'metadata': _safe_load(self.step_metadata) or {}
        }


def _get_env_knowledge_paths(agent_id: str, environment: str):
    env_dir = os.path.join("saas_agents", agent_id, "environments", environment)
    personalized = os.path.join(env_dir, f"{environment}_knowledge.json")
    default_template = os.path.join(env_dir, f"default_{environment}_knowledge.json")
    legacy = os.path.join(env_dir, "knowledge.json")
    return env_dir, personalized, default_template, legacy


def _read_json_file(path: str, fallback: Optional[Dict] = None) -> Dict:
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ Could not read JSON file {path}: {e}")
    return fallback.copy() if fallback else {}


def _write_json_file(path: str, data: Dict):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"❌ Could not write JSON file {path}: {e}")


def _load_environment_context(environment: str) -> Dict[str, Any]:
    if environment in ['pong', 'arena_pong']:
        from arena_pong_environment import ArenaPongEnvironment
        temp_env = ArenaPongEnvironment(match_id="context_builder")
        return temp_env.get_env_context()
    return {}


def _build_environment_block(environment: str, env_context: Dict[str, Any]) -> Dict[str, Any]:
    objective = env_context.get('objective', {})
    rules = env_context.get('rules', {})
    strategic = env_context.get('strategic_concepts', {})
    learning = env_context.get('learning_recommendations', {})
    reward_structure = {
        "ball_hit": 1.0,
        "ball_miss": -0.5,
        "score_point": 3.0,
        "concede_point": -0.5,
        "match_win": 10.0,
        "match_loss": -10.0
    }

    return {
        "environment_profile": {
            "environment_id": environment,
            "display_name": env_context.get('display_name', 'Arena Pong'),
            "environment_type": env_context.get('environment_type', 'competitive_real_time'),
            "understanding_level": "basic",
            "total_sessions": 0,
            "first_encountered": time.time(),
            "last_updated": time.time()
        },
        "objectives": {
            "primary": objective.get('primary', 'Score 21 points before opponent'),
            "secondary": objective.get('secondary', []),
            "victory_conditions": objective.get('victory_conditions', []),
            "failure_conditions": objective.get('failure_conditions', [])
        },
        "rules": {
            "core_mechanics": rules.get('core_mechanics', env_context.get('rules', [])),
            "constraints": rules.get('constraints', []),
            "scoring": rules.get('scoring', []),
            "special_conditions": rules.get('special_conditions', []),
            "game_mechanics": env_context.get('game_mechanics', {})
        },
        "strategic_framework": {
            "core_skills_required": strategic.get('core_skills', []),
            "tactical_approaches": strategic.get('tactical_approaches', []),
            "success_patterns": strategic.get('success_patterns', []),
            "failure_patterns": strategic.get('failure_patterns', []),
            "recommended_focus": strategic.get('recommended_focus', [])
        },
        "transferable_skills": env_context.get('transferable_skills', []),
        "learning_recommendations": learning,
        "reward_structure": reward_structure,
        "learning_parameters": {
            "learning_rate": 0.001,
            "exploration_rate": 0.3,
            "discount_factor": 0.99
        },
        "strategies": [],
        "lessons": [],
        "tactical_knowledge": [],
        "performance_patterns": [],
        "neural_insights": [],
        "knowledge_unlocks": env_context.get('knowledge_unlocks', []),
        "experiment_logs": []
    }


def _base_knowledge_document(agent_id: str, environment: str, env_context: Dict[str, Any]) -> Dict[str, Any]:
    env_block = _build_environment_block(environment, env_context)
    env_block['knowledge_progression'] = {
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
        "unlock_definitions": env_block.get('knowledge_unlocks', [])
    }

    return {
        "general_knowledge": {
            "transferable_strategies": [],
            "meta_learning_principles": [],
            "cross_environment_patterns": [],
            "abstract_concepts": [],
            "neural_symbolic_correlations": []
        },
        "environment_specific": {
            environment: env_block
        },
        "transfer_mappings": {
            "strategy_abstractions": {},
            "concept_translations": {},
            "success_patterns": [],
            "neural_pattern_mappings": {}
        },
        "symbolic_decision_history": [],
        "metadata": {
            "version": "2.3.0 - Environment Knowledge Refactor",
            "agent_id": agent_id,
            "environments": [environment],
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "transfer_learning_enabled": True,
            "neural_symbolic_integration": True,
            "environment_knowledge_integration": True
        }
    }


def _build_minimal_personalized_document(agent_id: str, environment: str,
                                         env_context: Dict[str, Any],
                                         default_doc: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    doc = deepcopy(default_doc) if default_doc else _base_knowledge_document(agent_id, environment, env_context)
    env_block = doc['environment_specific'][environment]

    env_block['objectives']['secondary'] = []
    env_block['objectives']['victory_conditions'] = env_block['objectives'].get('victory_conditions', [])[:1]
    env_block['objectives']['failure_conditions'] = env_block['objectives'].get('failure_conditions', [])[:1]

    core_mechanics = env_block['rules'].get('core_mechanics', [])
    env_block['rules']['core_mechanics'] = core_mechanics[:3] if core_mechanics else core_mechanics
    env_block['rules']['constraints'] = []
    env_block['rules']['scoring'] = []
    env_block['rules']['special_conditions'] = []

    env_block['strategic_framework']['core_skills_required'] = env_block['strategic_framework'].get(
        'core_skills_required', [])[:2]
    env_block['strategic_framework']['tactical_approaches'] = []
    env_block['strategic_framework']['success_patterns'] = []
    env_block['strategic_framework']['failure_patterns'] = []
    env_block['strategic_framework']['recommended_focus'] = env_block['strategic_framework'].get(
        'recommended_focus', [])[:1]

    env_block['transferable_skills'] = env_block.get('transferable_skills', [])
    env_block['learning_recommendations']['intermediate_focus'] = []
    env_block['learning_recommendations']['advanced_focus'] = []
    env_block['learning_recommendations']['expert_focus'] = []

    return doc


def _format_environment_payload(env_knowledge: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        'environment_profile': env_knowledge.get('environment_profile', {}),
        'objectives': env_knowledge.get('objectives', {}),
        'rules': env_knowledge.get('rules', {}),
        'strategic_framework': env_knowledge.get('strategic_framework', {}),
        'learning_recommendations': env_knowledge.get('learning_recommendations', {}),
        'reward_structure': env_knowledge.get('reward_structure', {}),
        'learning_parameters': env_knowledge.get('learning_parameters', {}),
        'custom_rewards': env_knowledge.get('custom_rewards', []),
        'custom_penalties': env_knowledge.get('custom_penalties', []),
        'knowledge_progression': env_knowledge.get('knowledge_progression', {}),
        'experiment_logs': env_knowledge.get('experiment_logs', [])
    }
    return payload


def _frontend_env_key(environment: str) -> str:
    return 'arena_pong' if environment == 'pong' else environment


def _ensure_environment_knowledge(agent_id: str, environment: str):
    env_dir, personalized_path, default_path, legacy_path = _get_env_knowledge_paths(agent_id, environment)
    os.makedirs(env_dir, exist_ok=True)
    env_context = _load_environment_context(environment if environment != 'pong' else 'arena_pong')

    if os.path.exists(legacy_path):
        if not os.path.exists(default_path):
            shutil.copy2(legacy_path, default_path)
        if not os.path.exists(personalized_path):
            shutil.copy2(legacy_path, personalized_path)
        try:
            os.remove(legacy_path)
        except OSError:
            pass

    default_doc = None
    if not os.path.exists(default_path):
        default_doc = _base_knowledge_document(agent_id, environment, env_context)
        _write_json_file(default_path, default_doc)
    else:
        default_doc = _read_json_file(default_path)

    if not os.path.exists(personalized_path):
        minimal_doc = _build_minimal_personalized_document(agent_id, environment, env_context, default_doc)
        _write_json_file(personalized_path, minimal_doc)

    personalized_doc = _read_json_file(personalized_path)
    default_doc = default_doc or _read_json_file(default_path)
    return personalized_path, default_path, personalized_doc, default_doc


# Enhanced GameSession class with Transfer Learning Support
class GameSession:
    """Enhanced game session with multi-environment transfer learning support"""

    def __init__(self, match_id: str, user_id: int, agent1_id: str, environment: str = 'pong'):
        self.match_id = match_id
        self.user_id = user_id
        self.agent1_id = agent1_id
        self.environment = environment
        self.room_id = f"match_{match_id}"
        self.trajectory_run_id: Optional[str] = None
        self.agent_config_snapshot_id: Optional[str] = None
        self.user_pref_snapshot_id: Optional[str] = None
        self.trajectory_buffer: List[Dict[str, Any]] = []

        # Initialize Arena environment with 256-dimension support
        if environment == 'pong':
            self.env = ArenaPongEnvironment(match_id=match_id, is_arena_match=True)
            self.adapter = PongArenaAdapter(self.env)
        else:
            # Future: Support for other environments
            self.env = ArenaPongEnvironment(match_id=match_id, is_arena_match=True)  # Default fallback
            self.adapter = PongArenaAdapter(self.env)
            print(f"⚠️ Environment {environment} not yet supported, using Arena Pong")

        # Initialize enhanced agent with transfer learning
        self.agent1 = self.load_enhanced_agent(agent1_id, environment)

        # Game state
        self.running = False
        self.game_thread = None
        self.match_start_time = None

        # Transfer learning tracking
        self.transfer_events = []
        self.transferable_skills_used = []
        self.knowledge_effectiveness = 0.0

        # Initialize Pong Decision Logger and Toolbox for knowledge sample tracking (Phase 4)
        if environment == 'pong':
            self.decision_logger = PongDecisionLogger()
            # Create toolbox wrapped around environment, connected to logger for tool tracking
            self.toolbox = PongToolbox(self.env, logger=self.decision_logger)
        else:
            self.decision_logger = None
            self.toolbox = None

        print(f"🎮 Enhanced game session created: {match_id} ({environment})")
        if self.agent1:
            envs = self.get_agent_environments(agent1_id)
            print(f"   🔄 Agent has experience in: {envs}")
        self._prepare_trajectory_run()

    def get_agent_environments(self, agent_id: str) -> List[str]:
        """Get list of environments agent has experience in"""
        try:
            with app.app_context():
                agent_record = Agent.query.get(agent_id)
                return agent_record.get_environments_experienced() if agent_record else []
        except:
            return []

    def load_enhanced_agent(self, agent_id: str, environment: str) -> Optional[AgentByte]:
        """Load enhanced Agent Byte with multi-environment support"""
        try:
            agent_record = Agent.query.get(agent_id)
            if not agent_record:
                raise ValueError(f"Agent {agent_id} not found")

            # Create enhanced Agent Byte with specific environment
            agent_byte = AgentByte(
                agent_id=agent_id,
                environment_id=environment,
                raw_state_size=14,  # Will be normalized to 256 internally
                action_size=3
            )

            # Set environment adapter for modular behavior
            agent_byte.set_environment(self.adapter)

            # Load any existing progress for this environment
            self._load_agent_progress(agent_byte, agent_record, environment)

            # Check for transfer learning opportunities
            self._check_transfer_opportunities(agent_byte, agent_record, environment)

            print(f"🧠 Enhanced agent loaded: {agent_id} for {environment}")
            return agent_byte

        except Exception as e:
            print(f"❌ Error loading enhanced agent {agent_id}: {e}")
            return None

    def _load_agent_progress(self, agent_byte: AgentByte, agent_record: Agent, environment: str):
        """Load existing progress for specific environment"""
        try:
            # The new agent structure handles loading automatically
            # through its file system in the constructor

            # Update agent with database stats
            agent_byte.games_played = agent_record.total_wins + agent_record.total_losses
            agent_byte.wins = agent_record.total_wins

            print(f"📊 Loaded progress: {agent_byte.games_played} games, {agent_byte.wins} wins")

        except Exception as e:
            print(f"⚠️ Error loading agent progress: {e}")

    def _check_transfer_opportunities(self, agent_byte: AgentByte, agent_record: Agent, target_environment: str):
        """Check for knowledge transfer opportunities"""
        try:
            experienced_envs = agent_record.get_environments_experienced()

            if len(experienced_envs) > 1:
                print(f"🔄 Transfer learning opportunity detected!")
                print(f"   Source environments: {[env for env in experienced_envs if env != target_environment]}")
                print(f"   Target environment: {target_environment}")

                # The agent will automatically detect and apply transferable knowledge
                # through its enhanced dual brain system

        except Exception as e:
            print(f"❌ Error checking transfer opportunities: {e}")

    def _prepare_trajectory_run(self):
        """Initialize trajectory run record and optional snapshots."""
        try:
            with app.app_context():
                match = Match.query.get(self.match_id)
                agent_record = Agent.query.get(self.agent1_id)

                if not match or not agent_record:
                    print("⚠️ Cannot prepare trajectory run without match/agent records")
                    return

                self.user_pref_snapshot_id = self._get_latest_user_pref_snapshot(match.user_id)

                config_snapshot = self._capture_agent_config_snapshot(agent_record)
                if config_snapshot:
                    self.agent_config_snapshot_id = config_snapshot.id

                run_metadata = {
                    'created_from': 'game_session',
                    'environment': self.environment,
                    'transfer_learning_ready': True
                }

                if hasattr(self.env, 'get_env_context'):
                    try:
                        run_metadata['env_context'] = self.env.get_env_context()
                    except Exception as ctx_err:
                        print(f"⚠️ Could not capture environment context for run: {ctx_err}")

                trajectory_run = TrajectoryRun(
                    match_id=self.match_id,
                    user_id=self.user_id,
                    agent_id=self.agent1_id,
                    environment=self.environment,
                    metadata=json.dumps(run_metadata),
                    agent_config_snapshot_id=self.agent_config_snapshot_id,
                    user_pref_snapshot_id=self.user_pref_snapshot_id
                )

                db.session.add(trajectory_run)
                db.session.commit()

                self.trajectory_run_id = trajectory_run.id
                self.trajectory_buffer = []
                print(f"🧾 Trajectory run initialized: {self.trajectory_run_id}")

        except Exception as e:
            db.session.rollback()
            print(f"⚠️ Could not prepare trajectory run: {e}")

    def _get_latest_user_pref_snapshot(self, user_id: int) -> Optional[str]:
        try:
            snapshot = UserPreferenceSnapshot.query.filter_by(user_id=user_id)\
                .order_by(UserPreferenceSnapshot.created_at.desc()).first()
            return snapshot.id if snapshot else None
        except Exception as e:
            print(f"⚠️ Could not fetch user preference snapshot: {e}")
            return None

    def _capture_agent_config_snapshot(self, agent_record: Agent) -> Optional[AgentConfigSnapshot]:
        try:
            runtime_config = {}
            decision_config = {}

            if self.agent1:
                runtime_config = {
                    'learning_rate': getattr(self.agent1, 'learning_rate', None),
                    'gamma': getattr(self.agent1, 'gamma', None),
                    'exploration_rate': getattr(self.agent1, 'exploration_rate', None),
                    'exploration_decay': getattr(self.agent1, 'exploration_decay', None),
                    'min_exploration': getattr(self.agent1, 'min_exploration', None),
                    'replay_batch_size': getattr(self.agent1, 'replay_batch_size', None),
                    'target_update_frequency': getattr(self.agent1, 'target_update_frequency', None),
                    'architecture_version': agent_record.architecture_version,
                    'environment': self.environment
                }

                decision_agent = getattr(self.agent1, 'decision_agent', None)
                if decision_agent:
                    decision_config = {
                        'learning_rate': getattr(decision_agent, 'learning_rate', None),
                        'gamma': getattr(decision_agent, 'gamma', None),
                        'exploration_rate': getattr(decision_agent, 'exploration_rate', None),
                        'exploration_decay': getattr(decision_agent, 'exploration_decay', None),
                        'min_exploration': getattr(decision_agent, 'min_exploration', None),
                        'batch_size': getattr(decision_agent, 'batch_size', None),
                        'target_update_frequency': getattr(decision_agent, 'target_update_frequency', None)
                    }

            config_snapshot = AgentConfigSnapshot(
                agent_id=self.agent1_id,
                match_id=self.match_id,
                config_data=json.dumps(runtime_config),
                decision_agent_config=json.dumps(decision_config)
            )

            db.session.add(config_snapshot)
            db.session.flush()
            return config_snapshot

        except Exception as e:
            db.session.rollback()
            print(f"⚠️ Could not capture agent config snapshot: {e}")
            return None

    def start_game(self):
        """Start game with enhanced transfer learning"""
        if self.running:
            return False

        # Check if agent loaded successfully
        if not self.agent1:
            print(f"❌ Cannot start game: Agent failed to load for {self.match_id}")
            return False

        print(f"🚀 Starting enhanced game: {self.match_id} ({self.environment})")

        self.running = True
        self.match_start_time = datetime.now(timezone.utc)

        # Reset environment
        self.env.reset_game()
        
        # Start the environment match
        self.env.start_match()

        # 🌍 Start enhanced agent training session with Arena Pong context
        env_context = self.env.get_env_context()
        match_id = f"match_{int(time.time())}"
        self.agent1.start_new_match(match_id, env_context)

        # Update match status
        match = Match.query.get(self.match_id)
        if match:
            match.status = 'active'
            match.started_at = self.match_start_time
            # Record that agent experienced this environment
            agent_record = Agent.query.get(self.agent1_id)
            if agent_record:
                agent_record.add_environment_experience(self.environment)
            db.session.commit()

        # Start game loop
        self.game_thread = threading.Thread(target=self._enhanced_game_loop, daemon=True)
        self.game_thread.start()

        # Send initial state
        self._send_game_state()
        return True

    def _enhanced_game_loop(self):
        """Enhanced game loop with transfer learning tracking"""
        try:
            print(f"🎯 Enhanced game loop started for {self.match_id}")

            while self.running:
                # Get current state for agent decision
                current_state = self.env.create_state()

                # Log decision step BEFORE toolbox calls (Phase 4: Prepare context and reset tracking)
                if self.decision_logger:
                    self.decision_logger.start_step(current_state, None)  # Action not yet determined

                # Get state using toolbox if available (tracks tool usage for knowledge logging)
                if self.toolbox:
                    # Use toolbox observation tools to track tool usage
                    ball_state = self.toolbox.read_ball_state()
                    paddles_state = self.toolbox.read_paddles_state()
                    score = self.toolbox.read_score()

                # Agent makes decision (with transfer learning)
                ai_action = self.agent1.get_action(current_state)

                # Update logger with actual action
                if self.decision_logger:
                    self.decision_logger.current_action = ai_action

                # Step environment using toolbox if available (tracks tool usage)
                if self.toolbox:
                    # Use toolbox action methods to execute action and track tool usage
                    if ai_action == 0:
                        next_state, reward, game_ended, info = self.toolbox.move_paddle_up()
                    elif ai_action == 2:
                        next_state, reward, game_ended, info = self.toolbox.move_paddle_down()
                    else:  # ai_action == 1
                        next_state, reward, game_ended, info = self.toolbox.hold_position()
                else:
                    # Fallback to direct environment call if toolbox not available
                    next_state, reward, game_ended, info = self.env.step(ai_action)

                # Log knowledge samples if meaningful event occurred (Phase 4)
                if self.decision_logger:
                    event = info.get("event", "none")
                    if event in ["ai_hit_ball", "ai_miss_ball"]:
                        self.decision_logger.log_step_result(event, reward, info)

                # Enhanced learning with transfer tracking
                self.agent1.learn(reward=reward, next_raw_state=next_state, done=game_ended)

                # Buffer trajectory data
                self._record_trajectory_step(
                    state=current_state,
                    next_state=next_state,
                    action=ai_action,
                    reward=reward,
                    done=game_ended or self.env.game_over
                )

                # Track transfer learning events
                self._track_transfer_events()

                # Send updated state
                self._send_game_state()

                # Check game end
                if game_ended or self.env.game_over:
                    self._handle_enhanced_game_end()
                    break

                time.sleep(1 / 30)  # 30 FPS

        except Exception as e:
            print(f"❌ Enhanced game loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False

    def _track_transfer_events(self):
        """Track transfer learning events during gameplay"""
        try:
            if hasattr(self.agent1, 'knowledge_transfer_events'):
                new_events = len(self.agent1.knowledge_transfer_events) - len(self.transfer_events)
                if new_events > 0:
                    self.transfer_events.extend(self.agent1.knowledge_transfer_events[-new_events:])

            if hasattr(self.agent1, 'transferable_skills_used'):
                new_skills = len(self.agent1.transferable_skills_used) - len(self.transferable_skills_used)
                if new_skills > 0:
                    self.transferable_skills_used.extend(self.agent1.transferable_skills_used[-new_skills:])

        except Exception as e:
            print(f"⚠️ Error tracking transfer events: {e}")

    def _record_trajectory_step(self, state, next_state, action, reward, done):
        """Buffer trajectory data for later persistence."""
        if not self.trajectory_run_id:
            return

        try:
            normalized_state = None
            normalized_next_state = None

            if self.agent1 and hasattr(self.agent1, 'network'):
                normalized_state = self.agent1.network.normalize_input(state) if state is not None else None
                normalized_next_state = self.agent1.network.normalize_input(next_state) if next_state is not None else None

            buffer_entry = {
                'timestamp': datetime.now(timezone.utc),
                'state': self._ensure_list(state),
                'next_state': self._ensure_list(next_state),
                'normalized_state': normalized_state.tolist() if normalized_state is not None else None,
                'normalized_next_state': normalized_next_state.tolist() if normalized_next_state is not None else None,
                'action': action,
                'reward': reward,
                'done': bool(done),
                'expert_choice': getattr(self.agent1, 'last_expert_choice', None),
                'rule_action': getattr(self.agent1, 'last_rule_action', None),
                'ddqn_action': getattr(self.agent1, 'last_ddqn_action', None),
                'metadata': {
                    'match_id': self.match_id
                }
            }

            self.trajectory_buffer.append(buffer_entry)
        except Exception as e:
            print(f"⚠️ Could not buffer trajectory step: {e}")

    def _finalize_trajectory_run(self, enhanced_stats: Optional[Dict[str, Any]] = None):
        """Persist buffered trajectory data to the database."""
        if not self.trajectory_run_id or not self.trajectory_buffer:
            return

        try:
            with app.app_context():
                run = TrajectoryRun.query.get(self.trajectory_run_id)
                if not run:
                    print(f"⚠️ Trajectory run {self.trajectory_run_id} missing; skipping persistence")
                    return

                total_reward = sum(step.get('reward', 0.0) for step in self.trajectory_buffer)
                run.total_steps = len(self.trajectory_buffer)
                run.total_reward = total_reward
                run.completed_at = datetime.now(timezone.utc)

                run.run_metadata = json.dumps({
                    'enhanced_stats': enhanced_stats or {},
                    'transfer_events': len(self.transfer_events),
                    'skills_used': len(self.transferable_skills_used)
                })

                for idx, step in enumerate(self.trajectory_buffer):
                    trajectory_step = TrajectoryStep(
                        run_id=self.trajectory_run_id,
                        step_index=idx,
                        timestamp=step['timestamp'],
                        state_vector=json.dumps(step['state']),
                        next_state_vector=json.dumps(step['next_state']),
                        normalized_state_vector=json.dumps(step['normalized_state']),
                        normalized_next_state_vector=json.dumps(step['normalized_next_state']),
                        action=step['action'],
                        reward=step['reward'],
                        done=step['done'],
                        expert_choice=step['expert_choice'],
                        rule_action=step['rule_action'],
                        ddqn_action=step['ddqn_action'],
                        step_metadata=json.dumps(step.get('metadata', {}))
                    )
                    db.session.add(trajectory_step)

                db.session.commit()
                print(f"💾 Persisted {len(self.trajectory_buffer)} trajectory steps for run {self.trajectory_run_id}")

        except Exception as e:
            db.session.rollback()
            print(f"❌ Error persisting trajectory buffer: {e}")
        finally:
            self.trajectory_buffer = []

    @staticmethod
    def _ensure_list(value):
        if value is None:
            return None
        if isinstance(value, list):
            return value
        if hasattr(value, 'tolist'):
            return value.tolist()
        try:
            return list(value)
        except Exception:
            return [value]

    def _execute_ai_action(self, action: int):
        """Execute AI action"""
        if action == 0:  # Move up
            self.env.ai_paddle_y -= self.env.paddle_speed
        elif action == 2:  # Move down
            self.env.ai_paddle_y += self.env.paddle_speed
        # action == 1 means stay

        # Keep paddle in bounds
        self.env.ai_paddle_y = max(0, min(self.env.height - self.env.paddle_height, self.env.ai_paddle_y))

    def _send_game_state(self):
        """Send enhanced game state with transfer learning info"""
        try:
            game_state = self.env.get_game_state()

            if self.agent1:
                agent_stats = self.agent1.get_stats()
                # Add transfer learning metrics
                if hasattr(self.agent1, 'get_transfer_readiness_report'):
                    transfer_report = self.agent1.get_transfer_readiness_report()
                    agent_stats['transfer_learning'] = {
                        'readiness_score': transfer_report.get('transfer_readiness_score', 0.0),
                        'environments_experienced': transfer_report.get('total_environments_experienced', 1),
                        'transferable_skills': transfer_report.get('transferable_skills_learned', 0)
                    }

                game_state['ai_stats'] = agent_stats

            game_state['match_type'] = 'user_vs_agent_enhanced'
            game_state['transfer_learning'] = {
                'events_count': len(self.transfer_events),
                'skills_used_count': len(self.transferable_skills_used),
                'environment': self.environment
            }

            socketio.emit('game_update', game_state, room=self.room_id)

        except Exception as e:
            print(f"❌ Error sending enhanced game state: {e}")

    def _handle_enhanced_game_end(self):
        """Handle game end with transfer learning analysis"""
        try:
            print(f"🏁 Enhanced game ended for {self.match_id}")

            # Determine winner
            winner_name = self.env.winner
            if winner_name == "Agent Byte":
                winner = 'agent1'
            else:
                winner = 'user'

            # Calculate transfer learning effectiveness
            self.knowledge_effectiveness = self._calculate_transfer_effectiveness()

            # Prepare final scores
            final_scores = {
                'agent1': self.env.ai_score,
                'user': self.env.player_score
            }

            # Get enhanced stats
            pong_stats = self.env.get_pong_stats()
            enhanced_stats = self.agent1.end_match(winner_name, final_scores, pong_stats)

            # Persist buffered trajectory data
            self._finalize_trajectory_run(enhanced_stats)

            # Update database with transfer learning metrics
            self._update_database_with_transfer_metrics(winner, final_scores, enhanced_stats)

            # Notify clients
            end_data = {
                'winner': winner,
                'final_scores': final_scores,
                'match_id': self.match_id,
                'match_duration': (datetime.now(timezone.utc) - self.match_start_time).total_seconds() / 60.0,
                'transfer_learning': {
                    'events_count': len(self.transfer_events),
                    'skills_used': len(self.transferable_skills_used),
                    'effectiveness': self.knowledge_effectiveness,
                    'environment': self.environment
                },
                'game_over': True
            }

            socketio.emit('game_ended', end_data, room=self.room_id)
            print(f"✅ Enhanced match completed with transfer learning analysis")

        except Exception as e:
            print(f"❌ Error handling enhanced game end: {e}")
        finally:
            # Ensure buffered data is flushed even if errors occurred
            if self.trajectory_buffer:
                self._finalize_trajectory_run()
            self.running = False

    def _calculate_transfer_effectiveness(self) -> float:
        """Calculate how effective transfer learning was in this match"""
        try:
            if not self.transfer_events and not self.transferable_skills_used:
                return 0.0

            # Simple effectiveness calculation based on agent performance
            match_reward = getattr(self.agent1, 'match_reward', 0)

            # Normalize reward to 0-1 scale
            effectiveness = max(0.0, min(1.0, (match_reward + 10) / 20))

            # Bonus for using transferable skills
            if self.transferable_skills_used:
                effectiveness = min(1.0, effectiveness + 0.1 * len(self.transferable_skills_used))

            return effectiveness

        except Exception as e:
            print(f"❌ Error calculating transfer effectiveness: {e}")
            return 0.0

    def _update_database_with_transfer_metrics(self, winner: str, final_scores: Dict, enhanced_stats: Dict):
        """Update database with transfer learning metrics"""
        try:
            with app.app_context():
                # Update match record
                match = Match.query.get(self.match_id)
                if match:
                    match.status = 'completed'
                    match.winner = winner
                    match.final_score = json.dumps(final_scores)
                    match.completed_at = datetime.now(timezone.utc)
                    match.transfer_events_count = len(self.transfer_events)
                    match.transferable_skills_used = len(self.transferable_skills_used)
                    match.knowledge_transfer_effectiveness = self.knowledge_effectiveness

                # Update agent record
                agent_record = Agent.query.get(self.agent1_id)
                if agent_record:
                    if winner == 'agent1':
                        agent_record.total_wins += 1
                    else:
                        agent_record.total_losses += 1

                    agent_record.last_trained = datetime.now(timezone.utc)
                    agent_record.total_training_time += 1.0

                    # Update transfer learning metrics
                    if hasattr(self.agent1, 'get_transfer_readiness_report'):
                        transfer_report = self.agent1.get_transfer_readiness_report()
                        agent_record.transferable_skills_count = transfer_report.get('transferable_skills_learned', 0)
                        agent_record.transfer_learning_maturity = transfer_report.get('transfer_readiness_score', 0.0)

                    # Update cross-environment performance
                    env_performance = {
                        'last_match_reward': getattr(self.agent1, 'match_reward', 0),
                        'transfer_effectiveness': self.knowledge_effectiveness,
                        'skills_used': len(self.transferable_skills_used),
                        'last_updated': time.time()
                    }
                    agent_record.update_cross_environment_performance(self.environment, env_performance)

                # Update user metrics
                user = User.query.get(self.user_id)
                if user:
                    user.total_training_time += 1.0
                    envs_before = user.total_environments_explored
                    # Check if this is a new environment for the user
                    user_agents = Agent.query.filter_by(user_id=self.user_id).all()
                    all_user_envs = set()
                    for agent in user_agents:
                        all_user_envs.update(agent.get_environments_experienced())
                    user.total_environments_explored = len(all_user_envs)

                db.session.commit()
                print(f"💾 Database updated with transfer learning metrics")

        except Exception as e:
            print(f"❌ Error updating database with transfer metrics: {e}")

    def move_player_paddle(self, direction: int):
        """Handle user paddle movement"""
        if not self.running:
            return

        self.env.move_player_paddle(direction)

        # Get user demo for enhanced learning
        demo_outcome = self.env.evaluate_user_action_outcome()
        if demo_outcome and self.agent1:
            self.agent1.record_user_demo(demo_outcome)

    def stop_game(self):
        """Stop the enhanced game"""
        print(f"🛑 Stopping enhanced game: {self.match_id}")
        self.running = False
        if self.game_thread:
            self.game_thread.join(timeout=2)


# Enhanced API Routes with Transfer Learning Support

@app.route('/')
def index():
    return render_template('base.html')


# Keep existing authentication routes (register, login, logout, get_current_user) unchanged

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')

        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400

        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 400

        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400

        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(email=email, password_hash=password_hash)

        db.session.add(user)
        db.session.commit()

        session['user_id'] = user.id
        session['email'] = user.email

        return jsonify({'success': True, 'user': user.to_dict()})

    except Exception as e:
        print(f"❌ Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500


@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')

        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400

        user = User.query.filter_by(email=email).first()

        if not user or not bcrypt.check_password_hash(user.password_hash, password):
            return jsonify({'error': 'Invalid email or password'}), 401

        session['user_id'] = user.id
        session['email'] = user.email

        return jsonify({'success': True, 'user': user.to_dict()})

    except Exception as e:
        print(f"❌ Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500


@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})


@app.route('/api/me')
def get_current_user():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({'user': user.to_dict()})


@app.route('/api/users/preferences', methods=['POST'])
def save_user_preferences():
    """Persist a new user preference snapshot for personalization-aware training."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    payload = request.get_json() or {}

    try:
        snapshot = UserPreferenceSnapshot(
            user_id=user_id,
            preferences=json.dumps({
                'control_scheme': payload.get('control_scheme', 'keyboard'),
                'playstyle': payload.get('playstyle', 'balanced'),
                'ui_theme': payload.get('ui_theme', 'default'),
                'difficulty': payload.get('difficulty', 'standard'),
                'notes': payload.get('notes')
            }),
            learning_goals=json.dumps(payload.get('learning_goals', [])),
            agent_focus=json.dumps(payload.get('agent_focus', [])),
            notes=payload.get('notes')
        )

        db.session.add(snapshot)
        db.session.commit()

        return jsonify({'success': True, 'snapshot': snapshot.to_dict()})

    except Exception as e:
        db.session.rollback()
        print(f"❌ Error saving user preferences: {e}")
        return jsonify({'error': 'Failed to save preferences'}), 500


@app.route('/api/users/preferences/latest', methods=['GET'])
def get_latest_user_preferences():
    """Return latest preference snapshot for the authenticated user."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        snapshot = UserPreferenceSnapshot.query.filter_by(user_id=user_id)\
            .order_by(UserPreferenceSnapshot.created_at.desc()).first()

        if not snapshot:
            return jsonify({'snapshot': None})

        return jsonify({'snapshot': snapshot.to_dict()})

    except Exception as e:
        print(f"❌ Error fetching user preferences: {e}")
        return jsonify({'error': 'Failed to fetch preferences'}), 500


# Enhanced Agent Management with Transfer Learning

@app.route('/api/agents', methods=['GET'])
def get_user_agents():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    agents = Agent.query.filter_by(user_id=user_id).order_by(Agent.created_at.desc()).all()
    agents_data = []

    for agent in agents:
        agent_dict = agent.to_dict()
        # Add transfer learning readiness assessment
        envs_count = len(agent.get_environments_experienced())
        transfer_readiness = "Ready for Transfer" if envs_count > 1 else "Single Environment"
        agent_dict['transfer_readiness'] = transfer_readiness
        agents_data.append(agent_dict)

    return jsonify({'agents': agents_data})


@app.route('/api/agents', methods=['POST'])
def create_agent():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        environment = data.get('environment', 'pong')

        if not name:
            return jsonify({'error': 'Agent name required'}), 400

        user = User.query.get(user_id)
        if user.subscription_tier == 'free' and len(user.agents) >= 5:  # Increased limit for v2.0
            return jsonify({'error': 'Free tier limited to 5 agents. Upgrade for unlimited agents.'}), 403

        # Create enhanced agent with multi-environment support
        agent = Agent(
            user_id=user_id,
            name=name,
            primary_environment=environment,
            environments_experienced=json.dumps([environment]),
            architecture_version='Agent Byte v2.0 - Transferable'
        )

        db.session.add(agent)

        # Update user stats
        user.total_agents_created += 1
        user.total_environments_explored = len(set(
            env for agent in user.agents + [agent]
            for env in agent.get_environments_experienced()
        ))

        db.session.commit()

        # Initialize agent file structure
        try:
            agent_dir = f"saas_agents/{agent.id}"
            os.makedirs(agent_dir, exist_ok=True)
            os.makedirs(f"{agent_dir}/core", exist_ok=True)
            os.makedirs(f"{agent_dir}/environments/{environment}", exist_ok=True)
            os.makedirs(f"{agent_dir}/transfers", exist_ok=True)
            print(f"📁 Created agent file structure: {agent_dir}")
            _ensure_environment_knowledge(agent.id, environment)
        except Exception as e:
            print(f"⚠️ Error creating agent file structure: {e}")

        return jsonify({'agent': agent.to_dict()})

    except Exception as e:
        print(f"❌ Create agent error: {e}")
        return jsonify({'error': 'Failed to create agent'}), 500


@app.route('/api/agents/<agent_id>/transfer-report', methods=['GET'])
def get_agent_transfer_report(agent_id):
    """Get detailed transfer learning report for an agent"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        agent = Agent.query.filter_by(id=agent_id, user_id=user_id).first()
        if not agent:
            return jsonify({'error': 'Agent not found'}), 404

        # Create enhanced agent instance to get transfer report
        try:
            agent_byte = AgentByte(
                agent_id=agent_id,
                environment_id=agent.primary_environment,
                raw_state_size=14,
                action_size=3
            )

            transfer_report = agent_byte.get_transfer_readiness_report()

            # Add database metrics
            transfer_report['database_metrics'] = {
                'total_wins': agent.total_wins,
                'total_losses': agent.total_losses,
                'environments_experienced': agent.get_environments_experienced(),
                'cross_environment_performance': agent.get_cross_environment_performance(),
                'elo_rating': agent.elo_rating
            }

            return jsonify({'transfer_report': transfer_report})

        except Exception as e:
            print(f"❌ Error generating transfer report: {e}")
            return jsonify({'error': 'Failed to generate transfer report'}), 500

    except Exception as e:
        print(f"❌ Transfer report error: {e}")
        return jsonify({'error': 'Failed to get transfer report'}), 500


@app.route('/api/agents/<agent_id>/environments', methods=['GET'])
def get_agent_environments(agent_id):
    """Get all environments an agent has experience in"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    agent = Agent.query.filter_by(id=agent_id, user_id=user_id).first()
    if not agent:
        return jsonify({'error': 'Agent not found'}), 404

    environments = agent.get_environments_experienced()
    cross_env_performance = agent.get_cross_environment_performance()

    environment_details = []
    for env in environments:
        env_data = {
            'name': env,
            'is_primary': env == agent.primary_environment,
            'performance': cross_env_performance.get(env, {}),
            'available': env == 'pong'  # Currently only Pong is available
        }
        environment_details.append(env_data)

    return jsonify({
        'environments': environment_details,
        'transfer_learning_enabled': len(environments) > 1
    })


# Agent Knowledge Management APIs

@app.route('/api/agents/<agent_id>/knowledge', methods=['GET'])
def get_agent_knowledge(agent_id):
    """Return personalized and default knowledge for editing."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        agent = Agent.query.filter_by(id=agent_id, user_id=user_id).first()
        if not agent:
            return jsonify({'error': 'Agent not found'}), 404

        environment = agent.primary_environment
        personalized_path, default_path, knowledge_data, default_doc = _ensure_environment_knowledge(agent_id, environment)
        env_key = _frontend_env_key(environment)

        env_knowledge = knowledge_data.get('environment_specific', {}).get(environment, {})
        default_env = default_doc.get('environment_specific', {}).get(environment, {})

        personalized_payload = _format_environment_payload(env_knowledge)
        default_payload = _format_environment_payload(default_env)

        return jsonify({
            'success': True,
            'knowledge_paths': {
                'personalized': personalized_path,
                'default': default_path
            },
            'knowledge': {'environment_specific': {env_key: personalized_payload}},
            'default_knowledge': {'environment_specific': {env_key: default_payload}},
            'agent_name': agent.name,
            'environment': environment
        })

    except Exception as e:
        print(f"❌ Error loading agent knowledge: {e}")
        return jsonify({'error': 'Failed to load agent knowledge'}), 500

@app.route('/api/agents/<agent_id>/knowledge', methods=['PUT'])
def update_agent_knowledge(agent_id):
    """Update personalized knowledge while keeping defaults intact."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        agent = Agent.query.filter_by(id=agent_id, user_id=user_id).first()
        if not agent:
            return jsonify({'error': 'Agent not found'}), 404

        payload = request.get_json() or {}
        env = agent.primary_environment
        env_key = _frontend_env_key(env)
        personalized_path, _, knowledge_data, _ = _ensure_environment_knowledge(agent_id, env)

        env_updates = payload.get('knowledge', {}).get('environment_specific', {}).get(env_key, {})
        env_store = knowledge_data.setdefault('environment_specific', {}).setdefault(env, {})

        preserved_progression = deepcopy(env_store.get('knowledge_progression', {}))
        preserved_unlocks = deepcopy(env_store.get('knowledge_unlocks', []))

        env_store.update(env_updates)
        env_store.pop('transferable_skills', None)

        if preserved_progression:
            env_store['knowledge_progression'] = preserved_progression
        if preserved_unlocks:
            env_store['knowledge_unlocks'] = preserved_unlocks

        knowledge_data.setdefault('metadata', {})['last_updated'] = datetime.now().isoformat()
        _write_json_file(personalized_path, knowledge_data)

        return jsonify({
            'success': True,
            'message': 'Agent basic instructions updated successfully'
        })

    except Exception as e:
        print(f"❌ Error updating agent basic instructions: {e}")
        return jsonify({'error': 'Failed to update agent basic instructions'}), 500

@app.route('/api/agents/<agent_id>/knowledge/reset', methods=['POST'])
def reset_agent_knowledge(agent_id):
    """Reset agent's knowledge to the default template for the environment."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        agent = Agent.query.filter_by(id=agent_id, user_id=user_id).first()
        if not agent:
            return jsonify({'error': 'Agent not found'}), 404

        env = agent.primary_environment
        personalized_path, default_path, _, default_doc = _ensure_environment_knowledge(agent_id, env)
        _write_json_file(personalized_path, default_doc)

        return jsonify({
            'success': True,
            'message': 'Agent knowledge reset to defaults'
        })

    except Exception as e:
        print(f"❌ Error resetting agent knowledge: {e}")
        return jsonify({'error': 'Failed to reset agent knowledge'}), 500

@app.route('/api/agents/<agent_id>/environment-defaults', methods=['GET'])
def get_environment_defaults(agent_id):
    """Return the default template knowledge for the agent's environment."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        agent = Agent.query.filter_by(id=agent_id, user_id=user_id).first()
        if not agent:
            return jsonify({'error': 'Agent not found'}), 404

        env = agent.primary_environment
        _, default_path, _, default_doc = _ensure_environment_knowledge(agent_id, env)
        env_key = _frontend_env_key(env)
        env_payload = _format_environment_payload(default_doc.get('environment_specific', {}).get(env, {}))

        return jsonify({
            'default_path': default_path,
            'environment_specific': {env_key: env_payload}
        })

    except Exception as e:
        print(f"❌ Error getting environment defaults: {e}")
        return jsonify({'error': 'Failed to get environment defaults'}), 500

@app.route('/api/agents/<agent_id>/load-environment-defaults', methods=['POST'])
def load_environment_defaults(agent_id):
    """Merge default template knowledge into the personalized file."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        agent = Agent.query.filter_by(id=agent_id, user_id=user_id).first()
        if not agent:
            return jsonify({'error': 'Agent not found'}), 404

        env = agent.primary_environment
        personalized_path, _, knowledge_data, default_doc = _ensure_environment_knowledge(agent_id, env)
        env_store = knowledge_data.setdefault('environment_specific', {}).setdefault(env, {})
        default_env = default_doc.get('environment_specific', {}).get(env, {})

        def merge_missing(dest, src):
            for key, value in src.items():
                if isinstance(value, dict):
                    merge_missing(dest.setdefault(key, {}), value)
                elif isinstance(value, list):
                    dest.setdefault(key, [])
                    for item in value:
                        if item not in dest[key]:
                            dest[key].append(item)
                else:
                    if key not in dest or dest[key] in (None, '', []):
                        dest[key] = value

        merge_missing(env_store, default_env)
        knowledge_data.setdefault('metadata', {})['last_updated'] = datetime.now().isoformat()
        _write_json_file(personalized_path, knowledge_data)

        return jsonify({
            'success': True,
            'message': 'Environment defaults merged successfully'
        })

    except Exception as e:
        print(f"❌ Error loading environment defaults: {e}")
        return jsonify({'error': 'Failed to load environment defaults'}), 500

@app.route('/api/matches', methods=['GET'])
def get_user_matches():
    """Get user's matches"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        matches = Match.query.filter_by(user_id=user_id).order_by(Match.started_at.desc()).limit(20).all()
        matches_data = [match.to_dict() for match in matches]
        return jsonify({'matches': matches_data})
    except Exception as e:
        print(f"❌ Error getting matches: {e}")
        return jsonify({'error': 'Failed to get matches'}), 500


@app.route('/api/matches', methods=['POST'])
def create_enhanced_match():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        data = request.get_json()
        agent_id = data.get('agent_id')
        environment = data.get('environment', 'pong')

        if not agent_id:
            return jsonify({'error': 'Agent ID required'}), 400

        agent = Agent.query.filter_by(id=agent_id, user_id=user_id).first()
        if not agent:
            return jsonify({'error': 'Agent not found'}), 404

        # Create enhanced match
        match = Match(
            user_id=user_id,
            agent1_id=agent_id,
            environment=environment,
            match_type='user_vs_agent_enhanced'
        )

        db.session.add(match)
        db.session.commit()

        return jsonify({'match': match.to_dict()})

    except Exception as e:
        print(f"❌ Create enhanced match error: {e}")
        return jsonify({'error': 'Failed to create match'}), 500


@app.route('/api/datasets/export', methods=['POST'])
def export_training_dataset():
    """Convert stored trajectories into a downloadable dataset artifact."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    payload = request.get_json() or {}
    agent_id = payload.get('agent_id')
    environment = payload.get('environment')

    try:
        if agent_id:
            agent = Agent.query.filter_by(id=agent_id, user_id=user_id).first()
            if not agent:
                return jsonify({'error': 'Agent not found'}), 404

        export_filters = {
            'user_id': user_id,
            'agent_id': agent_id,
            'environment': environment
        }

        dataset_info = build_and_export_dataset(
            db_session=db.session,
            trajectory_run_model=TrajectoryRun,
            trajectory_step_model=TrajectoryStep,
            preference_model=UserPreferenceSnapshot,
            config_model=AgentConfigSnapshot,
            output_dir=DATASET_EXPORT_DIR,
            filters=export_filters
        )

        if dataset_info['step_count'] == 0:
            return jsonify({'error': 'No trajectory data found for export'}), 404

        return jsonify({'success': True, 'dataset': dataset_info})

    except Exception as e:
        print(f"❌ Dataset export error: {e}")
        return jsonify({'error': 'Failed to export dataset'}), 500


# Enhanced WebSocket Events

@socketio.on('start_game')
def handle_start_enhanced_game(data):
    """Enhanced game start with transfer learning"""
    user_id = session.get('user_id')
    if not user_id:
        emit('error', {'message': 'Not authenticated'})
        return

    match_id = data.get('match_id')
    if not match_id:
        emit('error', {'message': 'Match ID required'})
        return

    try:
        match = Match.query.get(match_id)
        if not match:
            emit('error', {'message': 'Match not found'})
            return

        if match_id in active_games:
            emit('error', {'message': 'Game already active'})
            return

        # Create enhanced game session
        game_session = GameSession(match_id, user_id, match.agent1_id, match.environment)
        active_games[match_id] = game_session

        success = game_session.start_game()
        if success:
            emit('game_started', {
                'match_id': match_id,
                'environment': match.environment,
                'transfer_learning_enabled': True
            }, room=f"match_{match_id}")
            print(f"✅ Enhanced game started: {match_id}")
        else:
            del active_games[match_id]
            emit('error', {'message': 'Failed to start enhanced game'})

    except Exception as e:
        print(f"❌ Enhanced game start error: {e}")
        if match_id in active_games:
            del active_games[match_id]
        emit('error', {'message': f'Failed to start game: {str(e)}'})


# Keep other existing WebSocket handlers (join_match, move_paddle, stop_game, etc.)

@socketio.on('connect')
def handle_connect():
    user_id = session.get('user_id')
    if not user_id:
        emit('error', {'message': 'Not authenticated'})
        return False
    print(f"🔌 User {user_id} connected to enhanced platform")


@socketio.on('join_match')
def handle_join_match(data):
    user_id = session.get('user_id')
    if not user_id:
        emit('error', {'message': 'Not authenticated'})
        return

    match_id = data.get('match_id')
    room_id = f"match_{match_id}"
    join_room(room_id)
    emit('joined_match', {'match_id': match_id, 'enhanced': True})


@socketio.on('move_paddle')
def handle_move_paddle(data):
    match_id = data.get('match_id')
    direction = data.get('direction', 0)

    if match_id in active_games:
        active_games[match_id].move_player_paddle(direction)


@socketio.on('stop_game')
def handle_stop_game(data):
    match_id = data.get('match_id')
    if match_id in active_games:
        active_games[match_id].stop_game()
        del active_games[match_id]
        emit('game_stopped', {'match_id': match_id}, room=f"match_{match_id}")


# Database initialization
def init_enhanced_database():
    """Initialize enhanced database with transfer learning support"""
    with app.app_context():
        try:
            db.create_all()
            print("📊 Enhanced database tables created")

            # Create saas_agents directory structure
            base_dir = "../../saas_agents"
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
                print(f"📁 Created base directory: {base_dir}")

        except Exception as e:
            print(f"❌ Database initialization error: {e}")


# Background cleanup (enhanced)
def cleanup_finished_games():
    """Enhanced cleanup with transfer learning data preservation"""
    completed_games = []
    for match_id, game_session in active_games.items():
        if not game_session.running:
            completed_games.append(match_id)

    for match_id in completed_games:
        try:
            # Save any remaining transfer learning data before cleanup
            game_session = active_games[match_id]
            if hasattr(game_session, 'agent1') and game_session.agent1:
                game_session.agent1._save_enhanced_all_progress()

            del active_games[match_id]
            print(f"🧹 Cleaned up enhanced game session {match_id}")
        except Exception as e:
            print(f"❌ Error cleaning up session {match_id}: {e}")


def enhanced_periodic_tasks():
    """Enhanced periodic tasks with transfer learning maintenance"""
    while True:
        try:
            cleanup_finished_games()
            time.sleep(30)
        except Exception as e:
            print(f"❌ Enhanced periodic task error: {e}")


# Start enhanced background tasks
enhanced_periodic_thread = threading.Thread(target=enhanced_periodic_tasks, daemon=True)
enhanced_periodic_thread.start()

if __name__ == '__main__':
    print("🚀 Agent Byte SaaS Platform v2.0 - Multi-Environment Transfer Learning Starting...")
    print("=" * 80)
    print("🌐 Server: http://localhost:5000")
    print("🔧 Environment: Development")
    print("📊 Database: Enhanced SQLite with Transfer Learning Support")
    print("🎮 Features:")
    print("   ✅ Enhanced Agent Creation with Multi-Environment Support")
    print("   ✅ Transfer Learning Across Environments")
    print("   ✅ Standardized 256-Dimension Neural Networks")
    print("   ✅ Transferable Knowledge System")
    print("   ✅ Cross-Environment Performance Tracking")
    print("   ✅ Enhanced File Structure for Agent Persistence")
    print("   ✅ Transfer Learning Reports and Analytics")
    print("   ✅ Future-Ready for Multiple Game Environments")
    print("=" * 80)

    # Initialize enhanced database
    init_enhanced_database()

    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)