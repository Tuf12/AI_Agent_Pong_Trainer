# agent_arena.py - Fixed Agent vs Agent Battles with Universal Adapter Integration
import threading
import time
import json
import uuid
from datetime import datetime
from typing import Optional
import numpy as np

from arena_pong_environment import ArenaPongEnvironment
from agent_byte import AgentByte


class AgentArenaSession:
    """Enhanced Agent Arena Session with Universal Adapter Integration and Flask Context Management"""

    def __init__(self, match_id: str, user_id: int, agent1_id: str, agent2_id: str,
                 tournament_id: Optional[str] = None, is_public: bool = True,
                 socketio=None, db=None, app=None):
        self.match_id = match_id
        self.user_id = user_id
        self.agent1_id = agent1_id
        self.agent2_id = agent2_id
        self.tournament_id = tournament_id
        self.is_public = is_public
        self.socketio = socketio
        self.db = db
        self.app = app

        # Room management
        self.room_id = f"match_{match_id}"
        self.spectator_room = f"spectate_{match_id}"
        self.spectators = set()

        # Pure game environment (no AI dependencies)
        self.env = ArenaPongEnvironment(match_id=match_id, is_arena_match=True)

        # Enhanced Agent Byte instances with transfer learning
        self.agent1 = None
        self.agent2 = None

        # Game state
        self.running = False
        self.game_thread = None
        self.match_start_time = None

        # Performance tracking
        self.agent1_performance = {'score': 0, 'hits': 0, 'misses': 0}
        self.agent2_performance = {'score': 0, 'hits': 0, 'misses': 0}

        # Transfer learning tracking
        self.transfer_events = []
        self.dual_brain_decisions = []

        print(f"🤖 Enhanced Agent Arena Session created: {match_id}")

    def load_agents(self):
        """Load both agents using enhanced transfer learning system"""
        try:
            if self.app:
                with self.app.app_context():
                    return self._load_agents_with_context()
            else:
                raise RuntimeError("Flask app instance not provided - cannot access database")
        except Exception as e:
            print(f"❌ Error loading enhanced agents: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_agents_with_context(self):
        """Load agents within Flask context with enhanced transfer learning"""
        from saas_app import Agent

        # Load Agent 1
        agent1_record = self.db.session.get(Agent, self.agent1_id)
        if not agent1_record:
            raise ValueError(f"Agent 1 {self.agent1_id} not found")

        # Create enhanced Agent Byte with transferable intelligence
        self.agent1 = AgentByte(
            agent_id=self.agent1_id,
            environment_id='arena_pong',
            raw_state_size=14,  # Will be normalized to 256
            action_size=3
        )

        # Set environment for modular behavior
        self.agent1.set_environment(self.env)

        # Load Agent 2
        agent2_record = self.db.session.get(Agent, self.agent2_id)
        if not agent2_record:
            raise ValueError(f"Agent 2 {self.agent2_id} not found")

        # Create second enhanced Agent Byte
        self.agent2 = AgentByte(
            agent_id=self.agent2_id,
            environment_id='arena_pong',
            raw_state_size=14,
            action_size=3
        )

        # Set environment for modular behavior
        self.agent2.set_environment(self.env)

        # Load brain data and transfer learning progress
        self._load_agent_brain(self.agent1, agent1_record)
        self._load_agent_brain(self.agent2, agent2_record)

        print(f"✅ Enhanced agents loaded successfully!")
        print(f"   Agent 1: {agent1_record.name}")
        print(f"   Agent 2: {agent2_record.name}")
        print(f"   🧠🧩 Dual brain systems active")
        print(f"   🔄 Transfer learning enabled")

        return True

    def _load_agent_brain(self, agent_byte: AgentByte, agent_record):
        """Load agent brain data with enhanced transfer learning metrics"""
        if agent_record.brain_data:
            try:
                brain_data = json.loads(agent_record.brain_data)

                # Restore basic stats
                agent_byte.games_played = brain_data.get('games_played', 0)
                agent_byte.wins = brain_data.get('wins', 0)
                agent_byte.training_steps = brain_data.get('training_steps', 0)
                agent_byte.exploration_rate = brain_data.get('exploration_rate', 0.3)

                # Load transfer learning data if available
                if hasattr(agent_byte, 'transferable_skills_used'):
                    agent_byte.transferable_skills_used = brain_data.get('transferable_skills', [])

                if hasattr(agent_byte, 'knowledge_transfer_events'):
                    agent_byte.knowledge_transfer_events = brain_data.get('transfer_events', [])

                # Reduce exploration for arena battles (more confident play)
                if self.tournament_id:
                    agent_byte.exploration_rate = min(agent_byte.exploration_rate, 0.05)
                else:
                    agent_byte.exploration_rate = min(agent_byte.exploration_rate, 0.1)

                print(f"🧠 Enhanced brain loaded for {agent_record.name}:")
                print(f"   Games: {agent_byte.games_played}, Exploration: {agent_byte.exploration_rate:.3f}")
                print(f"   Transfer skills: {len(brain_data.get('transferable_skills', []))}")

            except Exception as e:
                print(f"⚠️ Error loading enhanced brain data for {agent_record.id}: {e}")

    def start_battle(self):
        """Start the enhanced agent vs agent battle"""
        if self.running:
            print(f"⚠️ Battle already running for {self.match_id}")
            return False

        print(f"🚀 Starting enhanced agent vs agent battle: {self.match_id}")

        # Load agents with enhanced transfer learning
        if not self.load_agents():
            print(f"❌ Failed to load enhanced agents for {self.match_id}")
            return False

        self.running = True
        self.match_start_time = datetime.utcnow()

        # Reset environment completely
        self.env.reset_game()
        self.env.start_match()

        # Reset performance tracking
        self.agent1_performance = {'score': 0, 'hits': 0, 'misses': 0}
        self.agent2_performance = {'score': 0, 'hits': 0, 'misses': 0}

        # Start enhanced match in both agents
        env_context = self._get_enhanced_env_context()
        self.agent1.start_new_match("arena_battle", env_context=env_context)
        self.agent2.start_new_match("arena_battle", env_context=env_context)

        # Update database with Flask context
        if self.db and self.app:
            with self.app.app_context():
                self._update_match_start()

        # Start enhanced game thread
        self.game_thread = threading.Thread(target=self._enhanced_battle_loop, daemon=True)
        self.game_thread.start()

        print(f"✅ Enhanced agent battle started successfully: {self.match_id}")
        return True

    def _get_enhanced_env_context(self) -> dict:
        """Get enhanced environment context with transfer learning info"""
        return {
            'environment_id': 'arena_pong',
            'environment_type': 'competitive_real_time',
            'objective': {'primary': 'Score 21 points before opponent in agent vs agent battle'},
            'rules': [
                'Hit ball with paddle to keep it in play',
                'Ball bounces off top/bottom walls',
                'Score when ball passes opponent paddle',
                'First to 21 points wins the match'
            ],
            'strategic_concepts': {
                'core_skills': [
                    'Ball trajectory prediction',
                    'Optimal paddle positioning',
                    'Timing optimization',
                    'Opponent pattern recognition'
                ],
                'tactical_approaches': [
                    'Predictive positioning for ball interception',
                    'Adaptive strategy based on opponent behavior',
                    'Error recovery after missed hits'
                ],
                'success_patterns': [
                    'Early positioning beats reactive movement',
                    'Consistent prediction over random actions',
                    'Adaptive timing based on ball velocity'
                ]
            },
            'transferable_skills': [
                'trajectory_prediction',
                'timing_optimization',
                'strategic_positioning',
                'pattern_recognition',
                'error_recovery',
                'adaptive_strategy'
            ],
            'learning_recommendations': {
                'neural_focus': ['Ball trajectory patterns', 'Paddle movement optimization'],
                'symbolic_focus': ['Opponent behavior patterns', 'Strategic positioning'],
                'transfer_focus': ['Universal prediction skills', 'Competitive strategy patterns']
            },
            'arena_context': {
                'match_type': 'agent_vs_agent',
                'is_tournament': bool(self.tournament_id),
                'is_public': self.is_public,
                'spectator_enabled': True
            }
        }

    def _update_match_start(self):
        """Update match start in database"""
        try:
            from saas_app import Match
            match = self.db.session.get(Match, self.match_id)
            if match:
                match.status = 'active'
                match.started_at = self.match_start_time
                self.db.session.commit()
        except Exception as e:
            print(f"❌ Error updating match start: {e}")

    def stop_battle(self):
        """Stop the enhanced battle"""
        print(f"🛑 Stopping enhanced battle: {self.match_id}")
        self.running = False
        if self.game_thread:
            self.game_thread.join(timeout=3)

    def _enhanced_battle_loop(self):
        """Enhanced battle loop with dual brain integration"""
        try:
            print(f"🎯 Enhanced battle loop started for {self.match_id}")
            step_count = 0

            while self.running and not self.env.game_over:
                step_count += 1

                # Get current game state
                current_state = self.env.create_state()

                # Agent 1 (AI paddle - right side) makes enhanced decision
                agent1_action = self.agent1.get_action(current_state)

                # Agent 2 (Player paddle - left side) makes enhanced decision with mirrored state
                mirrored_state = self._create_mirrored_state(current_state)
                agent2_action = self.agent2.get_action(mirrored_state)

                # Apply both agent movements
                self._apply_dual_agent_moves(agent1_action, agent2_action)

                # Process enhanced game physics with dual brain analysis
                reward1, reward2 = self._process_enhanced_battle_physics()

                # Enhanced learning with transfer tracking
                next_state = self.env.create_state()
                next_mirrored_state = self._create_mirrored_state(next_state)

                self.agent1.learn(reward=reward1, next_raw_state=next_state, done=self.env.game_over)
                self.agent2.learn(reward=reward2, next_raw_state=next_mirrored_state, done=self.env.game_over)

                # Track dual brain decisions every 10 steps
                if step_count % 10 == 0:
                    self._track_dual_brain_decisions(agent1_action, agent2_action, reward1, reward2)

                # Send enhanced game state to all clients
                game_state = self._get_enhanced_battle_state()
                if self.socketio:
                    self.socketio.emit('game_update', game_state, room=self.room_id)
                    self.socketio.emit('spectator_update', game_state, room=self.spectator_room)

                # Check if game ended
                if self.env.game_over:
                    threading.Thread(target=self._handle_enhanced_battle_end, daemon=True).start()
                    break

                # Control game speed (30 FPS)
                time.sleep(1 / 30)

        except Exception as e:
            print(f"❌ Enhanced battle loop error in {self.match_id}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            print(f"🏁 Enhanced battle loop ended for {self.match_id}")

    def _create_mirrored_state(self, state):
        """Create mirrored state for agent 2"""
        mirrored = state.copy()

        # Flip ball X position and velocity
        mirrored[0] = -state[0]  # Ball X: flip left-right
        mirrored[2] = -state[2]  # Ball X velocity: flip direction

        # Agent 2's position as "AI paddle" in its perspective
        agent2_paddle_center = self.env.player_paddle_y
        agent2_paddle_normalized = agent2_paddle_center / (self.env.height / 2)
        mirrored[4] = agent2_paddle_normalized

        # Flip distance calculations
        mirrored[5] = abs(-state[0] - (-1.0)) / 2.0

        return mirrored

    def _apply_dual_agent_moves(self, agent1_action, agent2_action):
        """Apply paddle movements from both agents"""
        # Agent 1 controls AI paddle (right side)
        movement1 = agent1_action - 1
        self.env.ai_paddle_y += movement1 * self.env.paddle_speed
        self.env.ai_paddle_y = max(-self.env.height / 2 + self.env.paddle_height / 2,
                                   min(self.env.height / 2 - self.env.paddle_height / 2, self.env.ai_paddle_y))

        # Agent 2 controls player paddle (left side)
        movement2 = agent2_action - 1
        self.env.player_paddle_y += movement2 * self.env.paddle_speed
        self.env.player_paddle_y = max(-self.env.height / 2 + self.env.paddle_height / 2,
                                       min(self.env.height / 2 - self.env.paddle_height / 2, self.env.player_paddle_y))

    def _process_enhanced_battle_physics(self):
        """Process ball physics and return enhanced rewards for both agents"""
        # Use the environment's step function but with dual agent context
        _, base_reward, _, _ = self.env.step(0)  # Dummy action since we handle movement separately

        # Calculate rewards based on actual game events
        reward1 = 0.0
        reward2 = 0.0

        # Agent 1 scoring
        if self.env.ai_score > self.agent1_performance['score']:
            reward1 += 3.0
            reward2 -= 0.5
            self.agent1_performance['score'] = self.env.ai_score

        # Agent 2 scoring
        if self.env.player_score > self.agent2_performance['score']:
            reward2 += 3.0
            reward1 -= 0.5
            self.agent2_performance['score'] = self.env.player_score

        # Hit tracking (simplified)
        if base_reward > 0:  # Ball was hit
            if self.env.ball_dx > 0:  # Ball moving toward agent 1
                reward1 += base_reward
                self.agent1_performance['hits'] += 1
            else:  # Ball moving toward agent 2
                reward2 += base_reward
                self.agent2_performance['hits'] += 1

        return reward1, reward2

    def _track_dual_brain_decisions(self, action1, action2, reward1, reward2):
        """Track dual brain decision making for analysis"""
        decision_record = {
            'timestamp': time.time(),
            'agent1_action': action1,
            'agent2_action': action2,
            'agent1_reward': reward1,
            'agent2_reward': reward2,
            'agent1_skills_used': len(getattr(self.agent1, 'transferable_skills_used', [])),
            'agent2_skills_used': len(getattr(self.agent2, 'transferable_skills_used', [])),
            'dual_brain_active': True
        }

        self.dual_brain_decisions.append(decision_record)

        # Keep recent history
        if len(self.dual_brain_decisions) > 100:
            self.dual_brain_decisions = self.dual_brain_decisions[-50:]

    def _get_enhanced_battle_state(self):
        """Get enhanced battle state with dual brain insights"""
        game_state = self.env.get_game_state()

        # Add agent statistics with transfer learning info
        if self.agent1:
            agent1_stats = self.agent1.get_stats() if hasattr(self.agent1, 'get_stats') else {}
            if hasattr(self.agent1, 'get_transfer_readiness_report'):
                transfer_report = self.agent1.get_transfer_readiness_report()
                agent1_stats['transfer_readiness'] = transfer_report.get('transfer_readiness_score', 0.0)
            game_state['agent1_stats'] = agent1_stats

        if self.agent2:
            agent2_stats = self.agent2.get_stats() if hasattr(self.agent2, 'get_stats') else {}
            if hasattr(self.agent2, 'get_transfer_readiness_report'):
                transfer_report = self.agent2.get_transfer_readiness_report()
                agent2_stats['transfer_readiness'] = transfer_report.get('transfer_readiness_score', 0.0)
            game_state['agent2_stats'] = agent2_stats

        game_state['match_type'] = 'enhanced_agent_vs_agent'
        game_state['spectator_count'] = len(self.spectators)
        game_state['agent1_performance'] = self.agent1_performance
        game_state['agent2_performance'] = self.agent2_performance
        game_state['dual_brain_decisions'] = len(self.dual_brain_decisions)
        game_state['transfer_learning_active'] = True

        return game_state

    def _handle_enhanced_battle_end(self):
        """Handle enhanced battle completion with transfer learning analysis"""
        try:
            print(f"🏁 Enhanced battle ending for {self.match_id}")

            winner = 'agent1' if self.env.ai_score > self.env.player_score else 'agent2'
            winner_name = "Agent 1" if winner == 'agent1' else "Agent 2"
            match_duration = (datetime.utcnow() - self.match_start_time).total_seconds() / 60.0

            final_scores = {'agent1': self.env.ai_score, 'agent2': self.env.player_score}

            # End enhanced agent matches with transfer learning analysis
            pong_stats = self.env.get_pong_stats()
            enhanced_stats1 = self.agent1.end_match(winner_name, final_scores, pong_stats)
            enhanced_stats2 = self.agent2.end_match(winner_name, final_scores, pong_stats)

            # Calculate transfer learning effectiveness
            transfer_effectiveness = self._calculate_transfer_effectiveness()

            # Update database with enhanced metrics
            if self.db and self.app:
                with self.app.app_context():
                    self._update_enhanced_match_database(winner, final_scores, match_duration, transfer_effectiveness)
                    self._update_enhanced_agent_stats(winner)
                    self._save_enhanced_agent_progress(self.agent1_id, self.agent1, enhanced_stats1)
                    self._save_enhanced_agent_progress(self.agent2_id, self.agent2, enhanced_stats2)

            # Notify clients with enhanced data
            end_data = {
                'winner': winner,
                'final_scores': final_scores,
                'match_id': self.match_id,
                'match_duration': round(match_duration, 1),
                'is_tournament_match': bool(self.tournament_id),
                'agent1_performance': self.agent1_performance,
                'agent2_performance': self.agent2_performance,
                'transfer_learning': {
                    'effectiveness': transfer_effectiveness,
                    'decisions_tracked': len(self.dual_brain_decisions),
                    'skills_demonstrated': self._get_skills_demonstrated()
                },
                'dual_brain_active': True,
                'game_over': True
            }

            if self.socketio:
                self.socketio.emit('game_ended', end_data, room=self.room_id)
                self.socketio.emit('spectator_game_ended', end_data, room=self.spectator_room)

            print(f"✅ Enhanced agent battle completed: {winner} ({winner_name}) wins!")
            print(f"   🧠🧩 Dual brain decisions: {len(self.dual_brain_decisions)}")
            print(f"   🔄 Transfer effectiveness: {transfer_effectiveness:.2f}")

        except Exception as e:
            print(f"❌ Error handling enhanced battle end: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False

    def _calculate_transfer_effectiveness(self) -> float:
        """Calculate how effective transfer learning was in this battle"""
        try:
            if not self.dual_brain_decisions:
                return 0.0

            # Analyze decision quality and reward correlation
            total_effectiveness = 0.0
            count = 0

            for decision in self.dual_brain_decisions:
                agent1_reward = decision.get('agent1_reward', 0)
                agent2_reward = decision.get('agent2_reward', 0)
                skills_used = decision.get('agent1_skills_used', 0) + decision.get('agent2_skills_used', 0)

                # Calculate effectiveness based on positive rewards and skill usage
                if skills_used > 0:
                    effectiveness = max(0, (agent1_reward + agent2_reward + 2) / 4)  # Normalize to 0-1
                    total_effectiveness += effectiveness
                    count += 1

            return total_effectiveness / max(1, count)

        except Exception as e:
            print(f"❌ Error calculating transfer effectiveness: {e}")
            return 0.0

    def _get_skills_demonstrated(self) -> list:
        """Get list of transferable skills demonstrated in this battle"""
        skills = set()

        if hasattr(self.agent1, 'transferable_skills_used'):
            for skill in self.agent1.transferable_skills_used:
                skills.add(skill.get('name', 'unknown_skill'))

        if hasattr(self.agent2, 'transferable_skills_used'):
            for skill in self.agent2.transferable_skills_used:
                skills.add(skill.get('name', 'unknown_skill'))

        return list(skills)

    def _update_enhanced_match_database(self, winner, final_scores, match_duration, transfer_effectiveness):
        """Update match record with enhanced transfer learning metrics"""
        try:
            from saas_app import Match
            match = self.db.session.get(Match, self.match_id)
            if match:
                match.status = 'completed'
                match.winner = winner
                match.final_score = json.dumps(final_scores)
                match.completed_at = datetime.utcnow()
                match.transfer_events_count = len(self.transfer_events)
                match.transferable_skills_used = len(self._get_skills_demonstrated())
                match.knowledge_transfer_effectiveness = transfer_effectiveness
                self.db.session.commit()
        except Exception as e:
            print(f"❌ Error updating enhanced match database: {e}")

    def _update_enhanced_agent_stats(self, winner):
        """Update agent win/loss stats with enhanced metrics"""
        try:
            from saas_app import Agent
            agent1_record = self.db.session.get(Agent, self.agent1_id)
            agent2_record = self.db.session.get(Agent, self.agent2_id)

            if agent1_record and agent2_record:
                if winner == 'agent1':
                    agent1_record.total_wins += 1
                    agent2_record.total_losses += 1
                else:
                    agent2_record.total_wins += 1
                    agent1_record.total_losses += 1

                # Update transfer learning metrics
                transfer_effectiveness = self._calculate_transfer_effectiveness()
                agent1_record.knowledge_transfer_success_rate = transfer_effectiveness
                agent2_record.knowledge_transfer_success_rate = transfer_effectiveness

                self.db.session.commit()
        except Exception as e:
            print(f"❌ Error updating enhanced agent stats: {e}")

    def _save_enhanced_agent_progress(self, agent_id, agent_byte, enhanced_stats):
        """Save enhanced agent progress with transfer learning data"""
        try:
            from saas_app import Agent
            agent_record = self.db.session.get(Agent, agent_id)
            if agent_record:
                # Enhanced brain data with transfer learning
                brain_data = {
                    'games_played': agent_byte.games_played,
                    'wins': agent_byte.wins,
                    'training_steps': agent_byte.training_steps,
                    'exploration_rate': agent_byte.exploration_rate,
                    'total_reward': getattr(agent_byte, 'total_reward', 0),
                    'transferable_skills': getattr(agent_byte, 'transferable_skills_used', []),
                    'transfer_events': getattr(agent_byte, 'knowledge_transfer_events', []),
                    'dual_brain_decisions': len(self.dual_brain_decisions),
                    'architecture': 'Agent Byte v2.0 - Enhanced Transfer Learning'
                }

                agent_record.brain_data = json.dumps(brain_data)
                agent_record.last_trained = datetime.utcnow()
                agent_record.total_training_time += 1.0

                # Update transfer learning maturity
                if hasattr(agent_byte, 'get_transfer_readiness_report'):
                    transfer_report = agent_byte.get_transfer_readiness_report()
                    agent_record.transfer_learning_maturity = transfer_report.get('transfer_readiness_score', 0.0)
                    agent_record.transferable_skills_count = transfer_report.get('transferable_skills_learned', 0)

                self.db.session.commit()
        except Exception as e:
            print(f"❌ Error saving enhanced agent progress: {e}")

    # Spectator management
    def add_spectator(self, user_id: Optional[int] = None):
        """Add a spectator to the enhanced battle"""
        spectator_id = f"user_{user_id}" if user_id else f"anon_{uuid.uuid4().hex[:8]}"
        self.spectators.add(spectator_id)
        print(f"👥 Spectator {spectator_id} joined enhanced battle {self.match_id}")
        return spectator_id

    def remove_spectator(self, spectator_id: str):
        """Remove a spectator from the enhanced battle"""
        self.spectators.discard(spectator_id)
        print(f"👥 Spectator {spectator_id} left enhanced battle {self.match_id}")


class SpectatorSession:
    """Enhanced spectator session with transfer learning insights"""

    def __init__(self, match_id: str, spectator_id: str, socketio=None):
        self.match_id = match_id
        self.spectator_id = spectator_id
        self.socketio = socketio

    def start_spectating(self):
        print(f"👀 Enhanced spectating started for {self.match_id}")
        return True

    def stop_spectating(self):
        print(f"👀 Enhanced spectating stopped for {self.match_id}")
        return True