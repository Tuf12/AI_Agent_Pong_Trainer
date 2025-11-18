# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

You are working on **Agent Byte v2.0**, an advanced multi-user SaaS platform for training transferable AI agents. This system features a sophisticated dual-brain architecture with standardized 256-dimension networks designed for cross-environment transfer learning.

**IMPORTANT**: This is NOT a simple Pong game. This is a cutting-edge transferable AI platform where agents learn skills that apply across multiple environments.

## Current Implementation Status: ADVANCED

### ✅ COMPLETED - Phases 1-3 Implementation
The core transferable AI architecture is **already implemented**:

1. **✅ Phase 1: Standardized Neural Network Architecture**
   - `StandardizedNetwork` class with 256→512→256→128→64→N architecture
   - Environment-agnostic core layers + environment-specific adapter layers
   - Network save/load with metadata and transfer learning tracking
   - Multi-environment file structure support

2. **✅ Phase 2: Enhanced Dual Brain System**
   - `DualBrainAgent` with multi-environment support
   - `AgentBrain` and `AgentKnowledge` with transfer learning capabilities
   - Cross-environment knowledge correlation and pattern recognition
   - Meta-learning principles tracking

3. **✅ Phase 3: Transferable Knowledge System**
   - `TransferableKnowledgeInterpreter` with universal strategies
   - Environment-agnostic strategy mappings (trajectory prediction, timing optimization, etc.)
   - Abstract lesson generation and cross-environment application
   - Symbolic decision making with transfer learning integration

### 🔄 CURRENT PHASE: Implementation Integration & Testing

## Advanced Architecture Overview

### Standardized Network Architecture
```python
# From agent_byte.py
class StandardizedNetwork:
    # 256 input → 512→256→128 (core) → 64 (adapter) → N (environment actions)
    def normalize_input(self, raw_state: np.ndarray) -> np.ndarray:
        # Converts any environment state to standard 256-dimension input
```

### Multi-Environment File Structure
```
saas_agents/{agent_id}/
├── core/
│   ├── agent_profile.json          # Agent identity, capabilities
│   ├── general_knowledge.json      # Abstract strategies
│   └── meta_learning.json          # Learning principles
├── environments/
│   ├── pong/
│   │   ├── network.npz             # Pong-specific network weights
│   │   ├── brain.json              # Neural learning state
│   │   ├── pong_knowledge.json     # Personalized Pong knowledge
│   │   ├── default_pong_knowledge.json  # Unlock template
│   │   └── matches.json            # Match history
│   └── {future_environment}/
└── transfers/
    ├── transfer_logs.json          # Knowledge transfer events
    └── network_evolution.json     # Cross-environment learning
```

### Transfer Learning Capabilities
- **Core Feature Learning**: Universal patterns (prediction, timing, adaptation)
- **Environment Adaptation**: Specific translations for each domain
- **Knowledge Transfer**: Abstract strategies apply across environments
- **Network Merging Foundation**: Architecture for future AI combination

## Key Implementation Files

### Core AI Components (FULLY IMPLEMENTED):
- **`agent_byte.py`**: 
  - `AgentByte` class with multi-environment support
  - `StandardizedNetwork` with 256-dim normalization
  - `MatchLogger` with environment-specific tracking
  - Transfer learning integration throughout

- **`dual_brain_system.py`**:
  - `DualBrainAgent` with cross-environment capabilities
  - `AgentBrain` with meta-learning principles
  - `AgentKnowledge` with transferable strategy storage
  - Knowledge migration and transfer systems

- **`knowledge_system.py`**:
  - `TransferableKnowledgeInterpreter` with universal strategies
  - `SymbolicDecisionMaker` with transfer learning support
  - Environment-agnostic situation analysis
  - Cross-domain strategy application

### SaaS Platform (ADVANCED IMPLEMENTATION):
- **`saas_app.py`**:
  - Multi-user agent creation with transfer learning
  - Enhanced game sessions with knowledge tracking
  - Transfer learning reports and analytics
  - Cross-environment performance monitoring

### Environment (NEEDS ATTENTION):
- **`pong_environment.py`**: ⚠️ **LEGACY - SHOULD BE REPLACED**
- **`arena_pong_environment.py`**: **MISSING - NEEDS CREATION**

## Development Commands

### Running the Application
```bash
# Start the SaaS platform
python saas_app.py

# The app runs on http://localhost:5000
# Uses Flask-SocketIO for real-time updates
# SQLite database: instance/agent_byte_saas.db
```

### Dependencies
The project uses standard Python libraries. Key dependencies include:
- Flask, Flask-SocketIO, Flask-SQLAlchemy, Flask-Bcrypt, Flask-CORS
- NumPy for neural network operations
- Standard library: json, threading, uuid, collections

### Testing & Code Quality
Currently no formal test suite. When implementing:
```bash
# Install development tools
pip install pytest flake8 black mypy

# Format code
black *.py

# Check style
flake8 *.py

# Type checking
mypy *.py

# Run tests (when created)
pytest
```

## Current Development Priorities

### 🎯 IMMEDIATE TASKS:

#### 1. Create Arena Pong Environment
**File**: `arena_pong_environment.py`
- Replace legacy `pong_environment.py` with SaaS-optimized version
- Integrate with standardized 256-dimension input system
- Support multi-user arena features (spectating, tournaments)
- Maintain modular symbolic interpretation methods

#### 2. SaaS Platform Integration Testing
- Verify multi-user agent isolation works correctly
- Test transfer learning effectiveness across sessions
- Validate file structure creation and management
- Ensure network save/load with metadata functions properly

#### 3. Transfer Learning Validation
- Test knowledge transfer between different agent instances
- Verify abstract strategies apply correctly in new contexts
- Monitor transfer learning effectiveness metrics
- Debug any cross-environment correlation issues

## Technical Implementation Details

### Agent Creation Flow:
```python
# Current advanced flow in saas_app.py
agent_byte = AgentByte(
    agent_id=agent_id,
    environment_id=environment,
    raw_state_size=14,  # Will be normalized to 256
    action_size=3
)

# Agent automatically:
# 1. Creates multi-environment file structure
# 2. Initializes standardized 256-dim network
# 3. Sets up transfer learning capabilities
# 4. Loads any existing cross-environment knowledge
```

### Transfer Learning Integration:
```python
# From knowledge_system.py - Universal strategies
"predict moving object trajectories": self._apply_trajectory_prediction,
"optimal timing execution": self._apply_timing_optimization,
"strategic positioning optimization": self._apply_strategic_positioning,
"pattern recognition and adaptation": self._apply_pattern_recognition,
```

### Network Architecture Standards:
- **Input**: 256 dimensions (standardized across ALL environments)
- **Core Layers**: [512, 256, 128] - Learn transferable patterns
- **Adapter Layer**: 64 dimensions - Environment-specific translation
- **Output Layer**: Variable per environment (3 for Pong)

## Advanced Features Available

### Transfer Learning Reports:
```python
# From agent_byte.py
def get_transfer_readiness_report(self) -> Dict[str, Any]:
    return {
        'transfer_readiness_score': float,
        'transferable_skills_learned': int,
        'cross_environment_experience': int,
        'recommended_next_environments': List[str]
    }
```

### Knowledge Transfer Analytics:
```python
# From dual_brain_system.py
def get_transfer_learning_report(self) -> Dict[str, Any]:
    return {
        'total_environments': int,
        'transfer_readiness': float,
        'meta_learning_maturity': float,
        'cross_environment_performance': Dict
    }
```

## Common Development Tasks

### Adding New Environment Support:
1. Create environment adapter with 256-dim state normalization
2. Define environment-specific rewards and action mappings
3. Update agent creation to support new environment
4. Add transfer learning context for cross-environment knowledge
5. Test knowledge transfer from existing environments

### Debugging Transfer Learning:
1. Check `agent.get_transfer_readiness_report()`
2. Monitor `transfer_events` during training
3. Verify `knowledge_transfer_effectiveness` metrics
4. Analyze cross-environment correlation patterns

### Performance Optimization:
1. Network operations are CPU-intensive - consider GPU support
2. Transfer learning adds computational overhead
3. File I/O for multi-environment structure
4. Real-time transfer learning analytics

## Critical Design Principles

### ✅ Maintain Transfer Learning Architecture:
- Core layers must remain environment-agnostic
- Adapter layers handle environment-specific translations
- Knowledge system maps abstract strategies to concrete actions
- File structure supports multi-environment agent evolution

### ✅ Preserve Modularity:
- Environments provide context, not storage management
- Core AI components remain domain-independent
- Transfer learning happens at the agent level, not environment level
- SaaS platform manages file paths and user isolation

### ❌ Avoid These Mistakes:
- Don't hardcode environment specifics in core AI components
- Don't break the standardized 256-dimension input requirement
- Don't bypass the transfer learning architecture
- Don't modify core/adapter layer separation

## Current File Status
- ✅ `agent_byte.py` - ADVANCED IMPLEMENTATION COMPLETE
- ✅ `dual_brain_system.py` - TRANSFER LEARNING COMPLETE  
- ✅ `knowledge_system.py` - UNIVERSAL STRATEGIES COMPLETE
- ✅ `saas_app.py` - MULTI-USER PLATFORM COMPLETE
- ❌ `pong_environment.py` - LEGACY, REMOVE AFTER REPLACEMENT
- ⏳ `arena_pong_environment.py` - **PRIORITY: NEEDS CREATION**

## Summary
You are working with a **highly advanced transferable AI system**. The core architecture is implemented and sophisticated. Focus on integration, testing, and creating the missing arena environment to complete the platform.