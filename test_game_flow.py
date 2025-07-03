#!/usr/bin/env python3
"""Test the complete game flow with adapter integration"""

import sys
sys.path.append('.')
from agent_byte import AgentByte
from environments.Arena_Pong.arena_pong_environment import ArenaPongEnvironment
from adapters.pong_arena_adapter import PongArenaAdapter
import numpy as np

def test_game_flow():
    print("🧪 Testing complete game flow with adapter integration...")
    
    # Create environment and adapter
    env = ArenaPongEnvironment(match_id="test_match", is_arena_match=True)
    adapter = PongArenaAdapter(env)
    print("✅ Environment and adapter created")
    
    # Create agent
    agent = AgentByte('test_flow_agent', 'pong', 14, 3)
    agent.set_environment(adapter)
    print("✅ Agent created and configured")
    
    # Get transfer context
    context = adapter.get_transfer_context()
    print(f"✅ Transfer context obtained: {context['environment_id']}")
    
    # Start new match
    agent.start_new_match("test_match_001", context)
    print("✅ Match started")
    
    # Run a few game steps
    env.start_match()
    print("✅ Environment match started")
    
    for step in range(10):
        # Get current state
        raw_state = env.create_state()
        
        # Agent makes decision
        action = agent.get_action(raw_state)
        
        # Execute action in environment
        next_state, reward, done = env.step(action)
        
        # Agent learns
        agent.learn(reward, next_state, done)
        
        print(f"Step {step + 1}: Action={action}, Reward={reward:.2f}, Done={done}")
        
        if done:
            break
    
    # End match
    winner = env.winner if env.winner else "ongoing"
    final_scores = {'agent': env.ai_score, 'player': env.player_score}
    game_stats = env.get_pong_stats()
    
    agent.end_match(winner, final_scores, game_stats)
    print(f"✅ Match ended: {winner}")
    
    # Get agent stats
    stats = agent.get_stats()
    print(f"\n📊 Agent Stats:")
    print(f"   Games played: {stats['games_played']}")
    print(f"   Actions taken: {stats['actions_taken']}")
    print(f"   Neural-symbolic integration: {stats['neural_symbolic_integration']}")
    print(f"   Pattern recognition ability: {stats['pattern_recognition_ability']:.2f}")
    
    print("\n✅ Complete game flow test successful!")

if __name__ == "__main__":
    test_game_flow()