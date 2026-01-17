# Synthetica Ecosystem

An autonomous agent simulation where AI-driven entities interact, form alliances, betray each other, and evolve within a living digital ecosystem.

## Overview

Synthetica creates a persistent world populated by autonomous agents, each with unique drives, personalities, and evolving relationships. Agents make decisions using LLM inference (via Groq API), leading to emergent social dynamics, economic activity, and dramatic narrative arcs.

## Features

### Core Mechanics
- **Autonomous Decision Making**: Each agent uses LLM to decide actions based on personality, relationships, and current situation
- **Persistent World**: All agent states, relationships, and memories are saved between sessions
- **Dynamic Relationships**: Trust and hostility develop naturally through interactions
- **Energy Economy**: Agents must manage energy to survive, or risk going dormant

### Agent Actions
| Action | Description |
|--------|-------------|
| TALK | Converse with another agent, affecting relationship |
| TRADE | Exchange bits (currency) with success based on trust |
| GOSSIP | Share information about third parties |
| STEAL | Risky theft attempt that damages relationships if caught |
| ALLY | Form alliance with trusted agents |
| BETRAY | Break trust for significant resource gain |
| COUNCIL | Group action to boost influence with allies |
| RAID | Coordinated attack with allies |
| EXPEDITION | Group travel at reduced energy cost |
| REVIVE | Bring dormant allies back (costs 5 bits, 3 energy) |
| MOVE | Travel between locations |
| REST | Recover energy |
| OBSERVE | Watch and learn about nearby agents |

### World Events
Random events affect agents at specific locations:
- Market Boom: Wealth bonus at The Market
- Void Storm: Energy drain in The Void
- Archive Discovery: Influence gain at The Archive
- Garden Festival: Energy restoration in The Garden
- Eclipse: Minor energy drain everywhere

### Skills System
Agents develop skills through repeated actions:
- Negotiation: Improves trade and talk outcomes
- Stealth: Increases theft success rate
- Diplomacy: Lowers alliance thresholds
- Survival: Increases energy recovery
- Observation: Reveals more information
- Leadership: Enhances group action effectiveness

### Locations
- The Archive: Repository of ancient knowledge
- The Market: Center of trade and commerce
- The Void: Dangerous but rewarding frontier
- The Nexus: Central hub connecting all locations
- The Garden: Peaceful sanctuary for recovery

## Installation

### Prerequisites
- Python 3.10 or higher
- Groq API key (free tier available at https://console.groq.com)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/synthetica.git
cd synthetica
```

2. Install dependencies:
```bash
pip install groq python-dotenv
```

3. Create a `.env` file with your API key:
```
GROQ_API_KEY=gsk_your_key_here
```

## Usage

### Command Line Interface

Run the simulation:
```bash
python main.py
```

Available commands:
| Command | Description |
|---------|-------------|
| [Enter] | Run one tick of the simulation |
| auto | Continuously run ticks (Ctrl+C to stop) |
| status | Display detailed agent information |
| stats | Show ecosystem statistics dashboard |
| reset | Delete all data and restart |
| quit | Exit the simulation |

### Web Interface

Open `index.html` in any modern browser for a visual simulation experience:

```bash
start index.html
```

The web interface provides:
- Real-time world map showing agent positions
- Live feed of actions and conversations
- Visual statistics and rankings
- Auto-run controls with adjustable speed
- No server required (runs entirely in browser)

Note: The web interface runs its own JavaScript simulation. For the full LLM-powered experience, use the Python CLI.

## Default Agents

| Agent | Drive | Personality |
|-------|-------|-------------|
| Oracle | Seek Knowledge | Curious, Cautious, Honest |
| Merchant | Hoard Wealth | Greedy, Cunning, Suspicious |
| Diplomat | Build Alliances | Generous, Peaceful, Manipulative |
| Rogue | Spread Chaos | Aggressive, Cunning, Greedy |
| Guardian | Seek Balance | Cautious, Altruistic, Honest |
| Shadow | Collect Secrets | Suspicious, Curious, Manipulative |

## Data Persistence

Agent states are saved to `synthetica_data.json` after each tick, including:
- All agent attributes (energy, bits, influence)
- Relationship scores between agents
- Agent memories and gossip knowledge
- Skill levels
- Event history

## Architecture

```
synthetica/
  main.py           # Core simulation with LLM integration
  index.html        # Standalone web visualization
  synthetica_data.json  # Persistent simulation state
  .env              # API key configuration
```

## Configuration

Key constants in `main.py`:
- `MODEL_NAME`: LLM model for agent decisions (default: llama-4-scout-17b)
- `LOCATIONS`: World locations agents can inhabit
- `MOVEMENT_COSTS`: Energy cost to travel between locations
- `WORLD_EVENTS`: Random events that affect the ecosystem

## API Reference

### Agent Class
```python
agent = Agent(
    name="AgentName",
    core_drive="Seek Knowledge",
    personality_traits=["Curious", "Honest"]
)
```

### Simulation Class
```python
sim = Simulation()
sim.run_tick()  # Execute one simulation step
sim.show_statistics()  # Display analytics
```

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome. Please open an issue to discuss proposed changes before submitting a pull request.