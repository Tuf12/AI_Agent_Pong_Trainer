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
from datetime import datetime, timezone
from typing import Dict, Optional, List, Any

# Import enhanced Agent Byte components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from agent_byte import AgentByte
from adapters.pong_arena_adapter import PongArenaAdapter

# Updated to use Arena Pong Environment v2.0 with 256-dimension support
from arena_pong_environment import ArenaPongEnvironment

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

# Global game sessions
active_games: Dict[str, 'GameSession'] = {}


# Enhanced Database Models with Multi-Environment Support
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
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



# Enhanced GameSession class with Transfer Learning Support
class GameSession:
    """Enhanced game session with multi-environment transfer learning support"""

    def __init__(self, match_id: str, user_id: int, agent1_id: str, environment: str = 'pong'):
        self.match_id = match_id
        self.user_id = user_id
        self.agent1_id = agent1_id
        self.environment = environment
        self.room_id = f"match_{match_id}"

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

        print(f"🎮 Enhanced game session created: {match_id} ({environment})")
        if self.agent1:
            envs = self.get_agent_environments(agent1_id)
            print(f"   🔄 Agent has experience in: {envs}")

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
                # Get current state
                current_state = self.env.create_state()

                # Agent makes decision (with transfer learning)
                ai_action = self.agent1.get_action(current_state)

                # Step environment (this handles action execution)
                next_state, reward, game_ended = self.env.step(ai_action)

                # Enhanced learning with transfer tracking
                self.agent1.learn(reward=reward, next_raw_state=next_state, done=game_ended)

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

            # Update database with transfer learning metrics
            self._update_database_with_transfer_metrics(winner, final_scores, enhanced_stats)

            # Notify clients
            end_data = {
                'winner': winner,
                'final_scores': final_scores,
                'match_id': self.match_id,
                'match_duration': (datetime.utcnow() - self.match_start_time).total_seconds() / 60.0,
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
                    match.completed_at = datetime.utcnow()
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

                    agent_record.last_trained = datetime.utcnow()
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
    """Get agent's knowledge data for editing - loads from environment context if needed"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        # Verify agent ownership
        agent = Agent.query.filter_by(id=agent_id, user_id=user_id).first()
        if not agent:
            return jsonify({'error': 'Agent not found'}), 404

        # Load agent's knowledge from file
        knowledge_file = f"saas_agents/{agent_id}/environments/{agent.primary_environment}/knowledge.json"
        
        # Get default Arena Pong context for complete knowledge
        from arena_pong_environment import ArenaPongEnvironment
        temp_env = ArenaPongEnvironment(match_id="temp_for_context")
        default_context = temp_env.get_env_context()
        
        if os.path.exists(knowledge_file):
            with open(knowledge_file, 'r') as f:
                knowledge_data = json.load(f)
        else:
            # Create complete knowledge structure from environment context
            knowledge_data = {
                "environment_specific": {
                    agent.primary_environment: {
                        "environment_profile": {
                            "environment_id": agent.primary_environment,
                            "display_name": default_context.get('display_name', 'Arena Pong'),
                            "environment_type": default_context.get('environment_type', 'competitive_real_time'),
                            "understanding_level": "basic",
                            "total_sessions": 0,
                            "first_encountered": time.time(),
                            "last_updated": time.time()
                        },
                        "objectives": {
                            "primary": default_context.get('objective', {}).get('primary', 'Score 21 points before opponent'),
                            "secondary": default_context.get('objective', {}).get('secondary', []),
                            "victory_conditions": default_context.get('objective', {}).get('victory_conditions', []),
                            "failure_conditions": default_context.get('objective', {}).get('failure_conditions', [])
                        },
                        "rules": {
                            "core_mechanics": default_context.get('rules', []),
                            "game_mechanics": default_context.get('game_mechanics', {})
                        },
                        "strategic_framework": {
                            "core_skills_required": default_context.get('strategic_concepts', {}).get('core_skills', []),
                            "tactical_approaches": default_context.get('strategic_concepts', {}).get('tactical_approaches', []),
                            "success_patterns": default_context.get('strategic_concepts', {}).get('success_patterns', []),
                            "failure_patterns": default_context.get('strategic_concepts', {}).get('failure_patterns', [])
                        },
                        "transferable_skills": default_context.get('transferable_skills', []),
                        "learning_recommendations": default_context.get('learning_recommendations', {}),
                        "reward_structure": {
                            "ball_hit": 1.0,
                            "ball_miss": -0.5,
                            "score_point": 3.0,
                            "concede_point": -0.5,
                            "match_win": 10.0,
                            "match_loss": -10.0
                        },
                        "learning_parameters": {
                            "learning_rate": 0.001,
                            "exploration_rate": 0.3,
                            "discount_factor": 0.99
                        }
                    }
                }
            }

        # Extract editable sections
        env_knowledge = knowledge_data.get('environment_specific', {}).get(agent.primary_environment, {})
        
        # Ensure all sections from environment context are present
        if not env_knowledge.get('objectives', {}).get('primary'):
            env_knowledge['objectives'] = {
                "primary": default_context.get('objective', {}).get('primary', 'Score 21 points before opponent'),
                "secondary": default_context.get('objective', {}).get('secondary', []),
                "victory_conditions": default_context.get('objective', {}).get('victory_conditions', []),
                "failure_conditions": default_context.get('objective', {}).get('failure_conditions', [])
            }
        
        if not env_knowledge.get('rules', {}).get('core_mechanics'):
            env_knowledge['rules'] = {
                "core_mechanics": default_context.get('rules', []),
                "game_mechanics": default_context.get('game_mechanics', {})
            }
        
        if not env_knowledge.get('strategic_framework', {}).get('core_skills_required'):
            env_knowledge['strategic_framework'] = {
                "core_skills_required": default_context.get('strategic_concepts', {}).get('core_skills', []),
                "tactical_approaches": default_context.get('strategic_concepts', {}).get('tactical_approaches', []),
                "success_patterns": default_context.get('strategic_concepts', {}).get('success_patterns', []),
                "failure_patterns": default_context.get('strategic_concepts', {}).get('failure_patterns', [])
            }
        
        if not env_knowledge.get('transferable_skills'):
            env_knowledge['transferable_skills'] = default_context.get('transferable_skills', [])
        
        if not env_knowledge.get('learning_recommendations'):
            env_knowledge['learning_recommendations'] = default_context.get('learning_recommendations', {})
        
        # Format knowledge data for frontend (which expects it under arena_pong)
        formatted_knowledge = {
            'environment_specific': {
                'arena_pong': {
                    'environment_profile': env_knowledge.get('environment_profile', {}),
                    'objectives': env_knowledge.get('objectives', {}),
                    'rules': env_knowledge.get('rules', {}),
                    'strategic_framework': env_knowledge.get('strategic_framework', {}),
                    'transferable_skills': env_knowledge.get('transferable_skills', []),
                    'learning_recommendations': env_knowledge.get('learning_recommendations', {}),
                    'reward_structure': env_knowledge.get('reward_structure', {}),
                    'learning_parameters': env_knowledge.get('learning_parameters', {}),
                    'custom_rewards': env_knowledge.get('custom_rewards', []),
                    'custom_penalties': env_knowledge.get('custom_penalties', [])
                }
            }
        }

        return jsonify({
            'success': True,
            'knowledge': formatted_knowledge,
            'agent_name': agent.name,
            'environment': agent.primary_environment
        })

    except Exception as e:
        print(f"❌ Error loading agent knowledge: {e}")
        return jsonify({'error': 'Failed to load agent knowledge'}), 500


@app.route('/api/agents/<agent_id>/knowledge', methods=['PUT'])
def update_agent_knowledge(agent_id):
    """Update agent's knowledge data with proper basic instructions handling"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        # Verify agent ownership
        agent = Agent.query.filter_by(id=agent_id, user_id=user_id).first()
        if not agent:
            return jsonify({'error': 'Agent not found'}), 404

        data = request.get_json()
        updated_knowledge = data.get('knowledge', {})

        # Load existing knowledge file
        knowledge_file = f"saas_agents/{agent_id}/environments/{agent.primary_environment}/knowledge.json"
        knowledge_dir = os.path.dirname(knowledge_file)
        os.makedirs(knowledge_dir, exist_ok=True)

        # Load or create knowledge structure
        if os.path.exists(knowledge_file):
            with open(knowledge_file, 'r') as f:
                knowledge_data = json.load(f)
        else:
            # Create basic structure if no file exists
            knowledge_data = {
                "general_knowledge": {
                    "transferable_strategies": [],
                    "meta_learning_principles": [],
                    "cross_environment_patterns": [],
                    "abstract_concepts": [],
                    "neural_symbolic_correlations": []
                },
                "environment_specific": {
                    agent.primary_environment: {
                        "environment_profile": {},
                        "objectives": {},
                        "rules": {},
                        "strategic_framework": {},
                        "transferable_skills": [],
                        "learning_recommendations": {},
                        "reward_structure": {},
                        "learning_parameters": {}
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
                    "version": "2.2.0 - Persistent Basic Instructions",
                    "agent_id": agent_id,
                    "environments": [agent.primary_environment],
                    "created": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "transfer_learning_enabled": True,
                    "neural_symbolic_integration": True,
                    "environment_knowledge_integration": True
                }
            }

        # Update the environment-specific knowledge with the full structure
        env_knowledge = knowledge_data['environment_specific'][agent.primary_environment]
        env_knowledge.update(updated_knowledge.get('environment_specific', {}).get('arena_pong', {}))

        # Ensure all basic instruction categories exist in knowledge structure
        if 'rules' not in env_knowledge:
            env_knowledge['rules'] = {}
        if 'core_mechanics' not in env_knowledge['rules']:
            env_knowledge['rules']['core_mechanics'] = []

        if 'strategic_framework' not in env_knowledge:
            env_knowledge['strategic_framework'] = {}
        for category in ['core_skills_required', 'success_patterns', 'failure_patterns', 'recommended_focus']:
            if category not in env_knowledge['strategic_framework']:
                env_knowledge['strategic_framework'][category] = []

        if 'transferable_skills' not in env_knowledge:
            env_knowledge['transferable_skills'] = []

        if 'objectives' not in env_knowledge:
            env_knowledge['objectives'] = {}

        # Update metadata
        knowledge_data['metadata']['last_updated'] = datetime.now().isoformat()

        # Save updated knowledge
        with open(knowledge_file, 'w') as f:
            json.dump(knowledge_data, f, indent=2)

        print(f"✅ Agent basic instructions updated: {agent_id}")
        return jsonify({
            'success': True,
            'message': 'Agent basic instructions updated successfully'
        })

    except Exception as e:
        print(f"❌ Error updating agent basic instructions: {e}")
        return jsonify({'error': 'Failed to update agent basic instructions'}), 500


@app.route('/api/agents/<agent_id>/knowledge/reset', methods=['POST'])
def reset_agent_knowledge(agent_id):
    """Reset agent's knowledge to ONLY environment defaults from get_env_context()"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        # Verify agent ownership
        agent = Agent.query.filter_by(id=agent_id, user_id=user_id).first()
        if not agent:
            return jsonify({'error': 'Agent not found'}), 404

        # Get default Arena Pong context
        from arena_pong_environment import ArenaPongEnvironment
        temp_env = ArenaPongEnvironment(match_id="temp_for_context")
        default_context = temp_env.get_env_context()

        # Create knowledge structure with ONLY environment defaults
        knowledge_data = {
            "general_knowledge": {
                "transferable_strategies": [],
                "meta_learning_principles": [],
                "cross_environment_patterns": [],
                "abstract_concepts": [],
                "neural_symbolic_correlations": []
            },
            "environment_specific": {
                agent.primary_environment: {
                    "environment_profile": {
                        "environment_id": agent.primary_environment,
                        "display_name": default_context.get('display_name', 'Arena Pong'),
                        "environment_type": default_context.get('environment_type', 'competitive_real_time'),
                        "understanding_level": "basic",
                        "total_sessions": 0,
                        "first_encountered": time.time(),
                        "last_updated": time.time()
                    },
                    # Primary objective from environment
                    "objectives": {
                        "primary": default_context.get('objective', {}).get('primary',
                                                                            'Score 21 points before opponent')
                    },
                    # Game rules from environment
                    "rules": {
                        "core_mechanics": default_context.get('rules', [])
                    },
                    # Strategic framework from environment
                    "strategic_framework": {
                        "core_skills_required": default_context.get('strategic_concepts', {}).get('core_skills', []),
                        "success_patterns": default_context.get('strategic_concepts', {}).get('success_patterns', []),
                        "failure_patterns": default_context.get('strategic_concepts', {}).get('failure_patterns', []),
                        "recommended_focus": default_context.get('learning_recommendations', {}).get('neural_focus', [])
                    },
                    # Transferable skills from environment
                    "transferable_skills": default_context.get('transferable_skills', []),

                    # Initialize empty structures for other data
                    "learning_recommendations": default_context.get('learning_recommendations', {}),
                    "reward_structure": {
                        "ball_hit": 1.0,
                        "ball_miss": -0.5,
                        "score_point": 3.0,
                        "concede_point": -0.5,
                        "match_win": 10.0,
                        "match_loss": -10.0
                    },
                    "learning_parameters": {
                        "learning_rate": 0.001,
                        "exploration_rate": 0.3,
                        "discount_factor": 0.99
                    },
                    "strategies": [],
                    "lessons": [],
                    "tactical_knowledge": [],
                    "performance_patterns": [],
                    "neural_insights": []
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
                "version": "2.2.0 - Reset to Environment Defaults",
                "agent_id": agent_id,
                "environments": [agent.primary_environment],
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "transfer_learning_enabled": True,
                "neural_symbolic_integration": True,
                "environment_knowledge_integration": True
            }
        }

        # Save reset knowledge
        knowledge_file = f"saas_agents/{agent_id}/environments/{agent.primary_environment}/knowledge.json"
        knowledge_dir = os.path.dirname(knowledge_file)
        os.makedirs(knowledge_dir, exist_ok=True)

        with open(knowledge_file, 'w') as f:
            json.dump(knowledge_data, f, indent=2)

        print(f"✅ Agent knowledge reset to environment defaults only: {agent_id}")
        return jsonify({
            'success': True,
            'message': 'Agent knowledge reset to environment defaults only'
        })

    except Exception as e:
        print(f"❌ Error resetting agent knowledge: {e}")
        return jsonify({'error': 'Failed to reset agent knowledge'}), 500


@app.route('/api/agents/<agent_id>/environment-defaults', methods=['GET'])
def get_environment_defaults(agent_id):
    """Get environment defaults for knowledge editor"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        agent = Agent.query.filter_by(id=agent_id, user_id=user_id).first()
        if not agent:
            return jsonify({'error': 'Agent not found'}), 404

        # Get environment from query params
        environment = request.args.get('environment', 'arena_pong')

        # Load environment defaults from Arena Pong Environment
        if environment == 'arena_pong':
            from arena_pong_environment import ArenaPongEnvironment

            # Create temporary environment to get context
            temp_env = ArenaPongEnvironment()
            env_context = temp_env.get_env_context()

            # Format for frontend - structured for basic instructions
            defaults = {
                'environment_id': environment,
                'objective': env_context.get('objective', {}),
                'rules': env_context.get('rules', []),
                'strategic_concepts': env_context.get('strategic_concepts', {}),
                'transferable_skills': env_context.get('transferable_skills', []),
                'learning_recommendations': env_context.get('learning_recommendations', {}),
                'performance_metrics': env_context.get('performance_metrics', {}),
                'success_indicators': env_context.get('success_indicators', {})
            }

            return jsonify(defaults)
        else:
            # Future: support other environments
            return jsonify({'error': f'Environment {environment} not supported yet'}), 400

    except Exception as e:
        print(f"❌ Error getting environment defaults: {e}")
        return jsonify({'error': 'Failed to get environment defaults'}), 500


@app.route('/api/agents/<agent_id>/load-environment-defaults', methods=['POST'])
def load_environment_defaults(agent_id):
    """Load environment defaults into agent knowledge (merge operation)"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        # Verify agent ownership
        agent = Agent.query.filter_by(id=agent_id, user_id=user_id).first()
        if not agent:
            return jsonify({'error': 'Agent not found'}), 404

        # Get environment context
        from arena_pong_environment import ArenaPongEnvironment
        temp_env = ArenaPongEnvironment()
        env_context = temp_env.get_env_context()

        # Load existing knowledge file
        knowledge_file = f"saas_agents/{agent_id}/environments/{agent.primary_environment}/knowledge.json"
        knowledge_dir = os.path.dirname(knowledge_file)
        os.makedirs(knowledge_dir, exist_ok=True)

        if os.path.exists(knowledge_file):
            with open(knowledge_file, 'r') as f:
                knowledge_data = json.load(f)
        else:
            return jsonify({'error': 'Agent knowledge not found'}), 404

        # Get current environment knowledge
        env_knowledge = knowledge_data["environment_specific"][agent.primary_environment]

        # Merge environment defaults with existing data (add missing defaults)
        def merge_defaults_with_existing(existing_list, default_list):
            """Add any missing defaults to existing list"""
            merged = list(existing_list) if existing_list else []
            for default_item in default_list:
                if default_item not in merged:
                    merged.append(default_item)
            return merged

        # Update with merged defaults
        env_knowledge["rules"]["core_mechanics"] = merge_defaults_with_existing(
            env_knowledge.get("rules", {}).get("core_mechanics", []),
            env_context.get('rules', [])
        )

        strategic_concepts = env_context.get('strategic_concepts', {})
        strategic_framework = env_knowledge.get("strategic_framework", {})

        strategic_framework["core_skills_required"] = merge_defaults_with_existing(
            strategic_framework.get("core_skills_required", []),
            strategic_concepts.get('core_skills', [])
        )
        strategic_framework["success_patterns"] = merge_defaults_with_existing(
            strategic_framework.get("success_patterns", []),
            strategic_concepts.get('success_patterns', [])
        )
        strategic_framework["failure_patterns"] = merge_defaults_with_existing(
            strategic_framework.get("failure_patterns", []),
            strategic_concepts.get('failure_patterns', [])
        )
        strategic_framework["recommended_focus"] = merge_defaults_with_existing(
            strategic_framework.get("recommended_focus", []),
            env_context.get('learning_recommendations', {}).get('neural_focus', [])
        )

        env_knowledge["strategic_framework"] = strategic_framework

        env_knowledge["transferable_skills"] = merge_defaults_with_existing(
            env_knowledge.get("transferable_skills", []),
            env_context.get('transferable_skills', [])
        )

        # Update metadata
        knowledge_data["metadata"]["last_updated"] = datetime.now().isoformat()

        # Save updated knowledge
        with open(knowledge_file, 'w') as f:
            json.dump(knowledge_data, f, indent=2)

        return jsonify({
            'success': True,
            'message': f'Environment defaults merged successfully'
        })

    except Exception as e:
        print(f"❌ Error loading environment defaults: {e}")
        return jsonify({'error': 'Failed to load environment defaults'}), 500

# Enhanced Match Management

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