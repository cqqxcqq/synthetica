"""
SYNTHETICA ECOSYSTEM - Enhanced Version
Run with: python main.py

Features:
- Event System: Random world events affecting agents
- Skills System: Agents develop skills through actions
- Group Actions: Multi-agent collaborative actions
- Statistics Dashboard: Rich analytics visualization
- Revival Mechanic: Allies can revive dormant agents
- Configuration Validation: Robust startup checks
- Agent Gossip: Information sharing between agents
"""

import os
import sys
import json
import random
import time
import uuid
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

# ============== STARTUP BANNER ==============
print("=" * 60)
print("🌐 THE SYNTHETICA ECOSYSTEM v2.0")
print("   Enhanced with Events, Skills, and Group Actions")
print("=" * 60)

# ============== ENVIRONMENT LOADING ==============
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


def validate_configuration() -> bool:
    """Validate all configuration before starting simulation."""
    errors = []
    warnings = []
    
    # Check API key
    if not GROQ_API_KEY:
        errors.append("No GROQ_API_KEY found! Create a .env file with: GROQ_API_KEY=gsk_your_key_here")
    elif not GROQ_API_KEY.startswith("gsk_"):
        warnings.append("API key doesn't start with 'gsk_' - might be invalid")
    
    # Check location graph connectivity
    all_locations = set()
    for loc_pair in MOVEMENT_COSTS.keys():
        all_locations.add(loc_pair[0])
        all_locations.add(loc_pair[1])
    
    missing_locations = set(LOCATIONS) - all_locations
    if missing_locations:
        warnings.append(f"Locations not connected: {missing_locations}")
    
    # Print results
    for warning in warnings:
        print(f"⚠ Warning: {warning}")
    
    for error in errors:
        print(f"❌ Error: {error}")
    
    if errors:
        return False
    
    print("✓ Configuration validated")
    return True


# ============== CONFIGURATION ==============
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
LOCATIONS = ["The Archive", "The Market", "The Void", "The Nexus", "The Garden"]
DATA_FILE = "synthetica_data.json"

MOVEMENT_COSTS = {
    ("The Archive", "The Market"): 2, ("The Market", "The Archive"): 2,
    ("The Archive", "The Void"): 3, ("The Void", "The Archive"): 3,
    ("The Market", "The Void"): 2, ("The Void", "The Market"): 2,
    ("The Nexus", "The Archive"): 1, ("The Archive", "The Nexus"): 1,
    ("The Nexus", "The Market"): 1, ("The Market", "The Nexus"): 1,
    ("The Nexus", "The Void"): 2, ("The Void", "The Nexus"): 2,
    ("The Garden", "The Nexus"): 1, ("The Nexus", "The Garden"): 1,
    ("The Garden", "The Archive"): 2, ("The Archive", "The Garden"): 2,
    ("The Garden", "The Market"): 2, ("The Market", "The Garden"): 2,
    ("The Garden", "The Void"): 3, ("The Void", "The Garden"): 3,
}

CORE_DRIVES = [
    "Hoard Wealth", "Seek Knowledge", "Spread Chaos", "Build Alliances",
    "Dominate Others", "Survive at All Costs", "Seek Balance", "Collect Secrets"
]

PERSONALITY_TRAITS = [
    "Suspicious", "Generous", "Cunning", "Naive", "Aggressive", "Peaceful",
    "Curious", "Cautious", "Manipulative", "Honest", "Greedy", "Altruistic"
]

# ============== NEW: WORLD EVENTS ==============
WORLD_EVENTS = [
    {
        "name": "Market Boom",
        "emoji": "📈",
        "description": "A surge of activity brings prosperity to The Market!",
        "effect": "bits",
        "delta": 3,
        "locations": ["The Market"]
    },
    {
        "name": "Void Storm",
        "emoji": "🌪️",
        "description": "A terrifying storm sweeps through The Void!",
        "effect": "energy",
        "delta": -3,
        "locations": ["The Void"]
    },
    {
        "name": "Archive Discovery",
        "emoji": "📜",
        "description": "Ancient knowledge surfaces in The Archive!",
        "effect": "influence",
        "delta": 4,
        "locations": ["The Archive"]
    },
    {
        "name": "Garden Festival",
        "emoji": "🌸",
        "description": "A peaceful festival restores energy to all in The Garden!",
        "effect": "energy",
        "delta": 4,
        "locations": ["The Garden"]
    },
    {
        "name": "Nexus Surge",
        "emoji": "⚡",
        "description": "The Nexus pulses with strange energy!",
        "effect": "random_skill",
        "delta": 2,
        "locations": ["The Nexus"]
    },
    {
        "name": "Eclipse",
        "emoji": "🌑",
        "description": "Darkness falls across the entire realm...",
        "effect": "energy",
        "delta": -1,
        "locations": LOCATIONS
    },
    {
        "name": "Merchant Caravan",
        "emoji": "🐪",
        "description": "A wealthy caravan passes through!",
        "effect": "bits",
        "delta": 2,
        "locations": ["The Market", "The Nexus"]
    },
    {
        "name": "Knowledge Rain",
        "emoji": "📚",
        "description": "Mysterious scrolls rain from the sky!",
        "effect": "random_skill",
        "delta": 1,
        "locations": ["The Archive", "The Garden"]
    }
]

# ============== NEW: SKILL DEFINITIONS ==============
SKILLS = {
    "negotiation": {"description": "Trading and talking effectiveness", "actions": ["TRADE", "TALK"]},
    "stealth": {"description": "Stealing success rate", "actions": ["STEAL"]},
    "diplomacy": {"description": "Alliance and betrayal outcomes", "actions": ["ALLY", "GOSSIP"]},
    "survival": {"description": "Energy management", "actions": ["REST", "REVIVE"]},
    "observation": {"description": "Information gathering", "actions": ["OBSERVE"]},
    "leadership": {"description": "Group action effectiveness", "actions": ["COUNCIL", "EXPEDITION", "RAID"]}
}

# Validate configuration early
if not validate_configuration():
    sys.exit(1)

# ============== GROQ CLIENT ==============
try:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    print("✓ Groq client ready")
except Exception as e:
    print(f"❌ Groq error: {e}")
    sys.exit(1)


# ============== AGENT CLASS (ENHANCED) ==============
class Agent:
    """Enhanced autonomous agent with skills and expanded capabilities."""
    
    def __init__(
        self,
        name: str,
        core_drive: str,
        personality_traits: List[str],
        agent_id: str = None,
        location: str = None,
        energy: int = 10,
        bits: int = 5,
        influence: int = 0,
        is_dormant: bool = False,
        relationships: Dict = None,
        memories: List = None,
        skills: Dict = None,
        gossip_knowledge: Dict = None,
        revival_count: int = 0
    ):
        self.id = agent_id or str(uuid.uuid4())[:8]
        self.name = name
        self.core_drive = core_drive
        self.personality_traits = personality_traits
        self.location = location or random.choice(LOCATIONS)
        self.energy = energy
        self.bits = bits
        self.influence = influence
        self.is_dormant = is_dormant
        self.relationships = relationships or {}
        self.memories = memories or []
        
        # NEW: Skills system
        self.skills = skills or {skill: 0 for skill in SKILLS.keys()}
        
        # NEW: Gossip knowledge (what agent knows about others' relationships)
        self.gossip_knowledge = gossip_knowledge or {}
        
        # NEW: Track revivals
        self.revival_count = revival_count

    def to_dict(self) -> Dict:
        """Serialize agent to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "core_drive": self.core_drive,
            "personality_traits": self.personality_traits,
            "location": self.location,
            "energy": self.energy,
            "bits": self.bits,
            "influence": self.influence,
            "is_dormant": self.is_dormant,
            "relationships": self.relationships,
            "memories": self.memories,
            "skills": self.skills,
            "gossip_knowledge": self.gossip_knowledge,
            "revival_count": self.revival_count
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Agent':
        """Deserialize agent from dictionary."""
        return cls(
            agent_id=data.get("id"),
            name=data["name"],
            core_drive=data["core_drive"],
            personality_traits=data["personality_traits"],
            location=data.get("location"),
            energy=data.get("energy", 10),
            bits=data.get("bits", 5),
            influence=data.get("influence", 0),
            is_dormant=data.get("is_dormant", False),
            relationships=data.get("relationships", {}),
            memories=data.get("memories", []),
            skills=data.get("skills", {skill: 0 for skill in SKILLS.keys()}),
            gossip_knowledge=data.get("gossip_knowledge", {}),
            revival_count=data.get("revival_count", 0)
        )

    def add_memory(self, memory: str, importance: int = 1):
        """Add memory with importance score for prioritization."""
        self.memories.append({
            "text": memory,
            "time": datetime.now().isoformat(),
            "importance": importance
        })
        # Keep last 30 memories, prioritizing important ones
        if len(self.memories) > 30:
            self.memories.sort(key=lambda m: m.get("importance", 1), reverse=True)
            self.memories = self.memories[:25]

    def get_recent_memories(self, n: int = 5) -> List[str]:
        """Get recent memories, prioritizing important ones."""
        sorted_memories = sorted(
            self.memories,
            key=lambda m: (m.get("importance", 1), m.get("time", "")),
            reverse=True
        )
        return [m["text"] for m in sorted_memories[:n]]

    def update_relationship(self, other_id: str, delta: int):
        """Update relationship with another agent, clamped to [-100, 100]."""
        current = self.relationships.get(other_id, 0)
        self.relationships[other_id] = max(-100, min(100, current + delta))

    def get_relationship(self, other_id: str) -> int:
        """Get relationship score with another agent."""
        return self.relationships.get(other_id, 0)

    def improve_skill(self, skill_name: str, amount: int = 1):
        """Improve a skill, capped at 100."""
        if skill_name in self.skills:
            self.skills[skill_name] = min(100, self.skills[skill_name] + amount)

    def get_skill_bonus(self, action: str) -> int:
        """Get bonus from skills for a given action."""
        bonus = 0
        for skill_name, skill_data in SKILLS.items():
            if action.upper() in skill_data["actions"]:
                bonus += self.skills.get(skill_name, 0) // 10  # 10 skill = +1 bonus
        return bonus

    def learn_gossip(self, about_id: str, from_id: str, relationship_score: int):
        """Learn gossip about another agent's relationship."""
        key = f"{from_id}:{about_id}"
        self.gossip_knowledge[key] = {
            "score": relationship_score,
            "learned_at": datetime.now().isoformat()
        }

    def get_top_skills(self, n: int = 3) -> List[Tuple[str, int]]:
        """Get agent's top skills."""
        sorted_skills = sorted(self.skills.items(), key=lambda x: x[1], reverse=True)
        return sorted_skills[:n]


# ============== SIMULATION CLASS (ENHANCED) ==============
class Simulation:
    """Enhanced simulation with events, group actions, and analytics."""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.tick_count = 0
        self.logs = []
        self.event_history = []
        self.load_data()

    def save_data(self):
        """Save simulation state to JSON file."""
        data = {
            "tick_count": self.tick_count,
            "agents": {aid: a.to_dict() for aid, a in self.agents.items()},
            "logs": self.logs[-200:],  # Keep more logs
            "event_history": self.event_history[-50:]
        }
        with open(DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    def load_data(self):
        """Load simulation state from JSON file."""
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, 'r') as f:
                    data = json.load(f)
                self.tick_count = data.get("tick_count", 0)
                self.logs = data.get("logs", [])
                self.event_history = data.get("event_history", [])
                for aid, adata in data.get("agents", {}).items():
                    self.agents[aid] = Agent.from_dict(adata)
                print(f"✓ Loaded {len(self.agents)} agents from save file")
            except Exception as e:
                print(f"⚠ Could not load save: {e}")

    def create_agent(self, name: str, core_drive: str, traits: List[str]) -> Agent:
        """Create and register a new agent."""
        agent = Agent(name=name, core_drive=core_drive, personality_traits=traits)
        self.agents[agent.id] = agent
        self.log(f"🆕 {name} joined the ecosystem at {agent.location}")
        return agent

    def initialize_default_agents(self):
        """Initialize the ecosystem with archetypal agents."""
        archetypes = [
            ("Oracle", "Seek Knowledge", ["Curious", "Cautious", "Honest"]),
            ("Merchant", "Hoard Wealth", ["Greedy", "Cunning", "Suspicious"]),
            ("Diplomat", "Build Alliances", ["Generous", "Peaceful", "Manipulative"]),
            ("Rogue", "Spread Chaos", ["Aggressive", "Cunning", "Greedy"]),
            ("Guardian", "Seek Balance", ["Cautious", "Altruistic", "Honest"]),
            ("Shadow", "Collect Secrets", ["Suspicious", "Curious", "Manipulative"]),
        ]
        for name, drive, traits in archetypes:
            self.create_agent(name, drive, traits)
        print(f"✓ Created {len(archetypes)} agents")

    def log(self, message: str, importance: int = 1):
        """Log an event with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.logs.append({"text": entry, "importance": importance})
        print(f"  {entry}")

    def get_agents_at_location(self, location: str) -> List[Agent]:
        """Get all active agents at a specific location."""
        return [a for a in self.agents.values() 
                if a.location == location and not a.is_dormant]

    def get_active_agents(self) -> List[Agent]:
        """Get all non-dormant agents."""
        return [a for a in self.agents.values() if not a.is_dormant]

    def get_dormant_agents(self) -> List[Agent]:
        """Get all dormant agents."""
        return [a for a in self.agents.values() if a.is_dormant]

    # ============== NEW: EVENT SYSTEM ==============
    def trigger_random_event(self) -> Optional[Dict]:
        """Randomly trigger a world event (15% chance)."""
        if random.random() > 0.15:
            return None
        
        event = random.choice(WORLD_EVENTS)
        affected_agents = []
        
        print(f"\n  {event['emoji']} WORLD EVENT: {event['name']}!")
        print(f"     {event['description']}")
        
        for agent in self.get_active_agents():
            if agent.location in event["locations"]:
                affected_agents.append(agent.name)
                
                if event["effect"] == "bits":
                    agent.bits = max(0, agent.bits + event["delta"])
                elif event["effect"] == "energy":
                    agent.energy = max(0, min(20, agent.energy + event["delta"]))
                elif event["effect"] == "influence":
                    agent.influence += event["delta"]
                elif event["effect"] == "random_skill":
                    skill = random.choice(list(SKILLS.keys()))
                    agent.improve_skill(skill, event["delta"])
                
                agent.add_memory(f"Witnessed {event['name']}: {event['description']}", importance=3)
        
        if affected_agents:
            print(f"     Affected: {', '.join(affected_agents)}")
        
        self.event_history.append({
            "tick": self.tick_count,
            "event": event["name"],
            "affected": affected_agents,
            "time": datetime.now().isoformat()
        })
        
        self.log(f"{event['emoji']} {event['name']} affected {len(affected_agents)} agents", importance=3)
        return event

    # ============== NEW: STATISTICS DASHBOARD ==============
    def show_statistics(self):
        """Display rich ecosystem statistics."""
        print("\n" + "=" * 60)
        print("📊 ECOSYSTEM STATISTICS")
        print("=" * 60)
        
        active = self.get_active_agents()
        dormant = self.get_dormant_agents()
        
        # Basic stats
        print(f"\n📈 POPULATION: {len(active)} active, {len(dormant)} dormant")
        print(f"🕐 TOTAL TICKS: {self.tick_count}")
        print(f"🌍 WORLD EVENTS: {len(self.event_history)}")
        
        # Location distribution
        print("\n📍 LOCATION DISTRIBUTION:")
        for loc in LOCATIONS:
            agents_here = [a for a in active if a.location == loc]
            bar = "█" * len(agents_here) + "░" * (6 - len(agents_here))
            names = ", ".join([a.name for a in agents_here]) or "empty"
            print(f"   {loc:15} [{bar}] {len(agents_here)} - {names}")
        
        # Wealth distribution
        print("\n💰 WEALTH RANKING:")
        sorted_by_bits = sorted(self.agents.values(), key=lambda a: a.bits, reverse=True)
        max_bits = max(a.bits for a in self.agents.values()) if self.agents else 1
        for i, agent in enumerate(sorted_by_bits[:6]):
            bar_len = int((agent.bits / max_bits) * 20) if max_bits > 0 else 0
            bar = "█" * bar_len + "░" * (20 - bar_len)
            status = "💀" if agent.is_dormant else "✓"
            rank = "👑" if i == 0 else f"{i+1}."
            print(f"   {rank} {status} {agent.name:12} [{bar}] {agent.bits} bits")
        
        # Influence ranking
        print("\n⭐ INFLUENCE RANKING:")
        sorted_by_influence = sorted(self.agents.values(), key=lambda a: a.influence, reverse=True)
        for i, agent in enumerate(sorted_by_influence[:3]):
            print(f"   {i+1}. {agent.name}: {agent.influence} influence")
        
        # Skill leaders
        print("\n🎯 SKILL LEADERS:")
        for skill_name in SKILLS.keys():
            best = max(self.agents.values(), key=lambda a: a.skills.get(skill_name, 0))
            if best.skills.get(skill_name, 0) > 0:
                print(f"   {skill_name.capitalize():12} - {best.name} ({best.skills[skill_name]})")
        
        # Relationship network
        print("\n🤝 RELATIONSHIP HIGHLIGHTS:")
        all_relationships = []
        for agent in self.agents.values():
            for other_id, score in agent.relationships.items():
                if other_id in self.agents:
                    other = self.agents[other_id]
                    all_relationships.append((agent.name, other.name, score))
        
        if all_relationships:
            # Strongest alliance
            best = max(all_relationships, key=lambda r: r[2])
            if best[2] > 0:
                print(f"   💚 Strongest Alliance: {best[0]} ↔ {best[1]} (+{best[2]})")
            
            # Worst conflict
            worst = min(all_relationships, key=lambda r: r[2])
            if worst[2] < 0:
                print(f"   💔 Worst Conflict: {worst[0]} ↔ {worst[1]} ({worst[2]})")
        
        # Recent events
        if self.event_history:
            print("\n🌍 RECENT EVENTS:")
            for event in self.event_history[-3:]:
                print(f"   Tick {event['tick']}: {event['event']}")
        
        print("\n" + "=" * 60)

    # ============== PROMPT BUILDER (ENHANCED) ==============
    def build_prompt(self, agent: Agent, others: List[Agent]) -> str:
        """Build the decision prompt for an agent with enhanced actions."""
        others_text = ""
        for o in others:
            rel = agent.get_relationship(o.id)
            rel_word = "neutral"
            if rel > 60: rel_word = "allied"
            elif rel > 30: rel_word = "friendly"
            elif rel < -60: rel_word = "enemy"
            elif rel < -30: rel_word = "hostile"
            others_text += f"\n  - {o.name}: {rel_word} (score: {rel})"
        
        if not others_text:
            others_text = "\n  No one else here."

        memories_text = "\n  ".join(agent.get_recent_memories(3)) or "No memories yet."
        
        # Movement options
        move_options = []
        for loc in LOCATIONS:
            if loc != agent.location:
                cost = MOVEMENT_COSTS.get((agent.location, loc), 3)
                if agent.energy >= cost:
                    move_options.append(f"{loc} (cost: {cost})")

        # Skills summary
        top_skills = agent.get_top_skills(3)
        skills_text = ", ".join([f"{s[0]}:{s[1]}" for s in top_skills if s[1] > 0]) or "None developed"

        # Check for dormant agents that could be revived
        dormant_at_loc = [a for a in self.get_dormant_agents() 
                         if a.location == agent.location and agent.get_relationship(a.id) > 40]
        
        # Check for allies for group actions
        allies_here = [o for o in others if agent.get_relationship(o.id) > 30]

        return f"""You are {agent.name}, an autonomous agent in the Synthetica Ecosystem.

YOUR IDENTITY:
- Core Drive: {agent.core_drive}
- Personality: {', '.join(agent.personality_traits)}
- Top Skills: {skills_text}

CURRENT STATUS:
- Location: {agent.location}
- Energy: {agent.energy}/20
- Bits (wealth): {agent.bits}
- Influence: {agent.influence}

AGENTS PRESENT:{others_text}

ALLIES HERE: {', '.join([a.name for a in allies_here]) or 'None'}

YOUR RECENT MEMORIES:
  {memories_text}

AVAILABLE ACTIONS:
SOCIAL:
- TALK [name]: Have a conversation (improves negotiation skill)
- TRADE [name]: Propose resource exchange
- GOSSIP [name]: Share relationship info with someone (improves diplomacy)
- ALLY [name]: Propose alliance (needs positive relationship)

AGGRESSIVE:
- STEAL [name]: Attempt theft (risky! improves stealth)
- BETRAY [name]: Betray someone who trusts you

GROUP ACTIONS (need allies present):
- COUNCIL: Form a council with allies to boost influence
- RAID [name]: Coordinate attack with allies
- EXPEDITION [location]: Propose group travel

UTILITY:
- MOVE [location]: Travel somewhere. Options: {', '.join(move_options) if move_options else 'None (low energy)'}
- REST: Recover +3 energy (improves survival)
- OBSERVE: Watch and learn about others (improves observation)
{f'- REVIVE [name]: Revive a dormant ally (cost: 5 bits, 3 energy). Available: {", ".join([a.name for a in dormant_at_loc])}' if dormant_at_loc else ''}

Based on your drive "{agent.core_drive}" and personality, decide your action.

Respond in this EXACT JSON format:
{{"action": "ACTION_NAME", "target": "name_or_location_or_null", "reasoning": "why you chose this", "speech": "what you say out loud or null"}}
"""

    def get_llm_decision(self, agent: Agent, prompt: str) -> Dict:
        """Get decision from LLM with robust error handling."""
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an autonomous agent making decisions. Always respond with valid JSON only, no extra text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.85,
                max_tokens=300
            )
            
            text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            # Find JSON object
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]
            
            return json.loads(text)
            
        except json.JSONDecodeError:
            return {"action": "OBSERVE", "target": None, "reasoning": "Thinking...", "speech": None}
        except Exception as e:
            print(f"    ⚠ LLM Error: {e}")
            return {"action": "REST", "target": None, "reasoning": str(e), "speech": None}

    def find_agent_by_name(self, name: str, agents: List[Agent] = None) -> Optional[Agent]:
        """Find an agent by name (case-insensitive)."""
        if not name:
            return None
        name_lower = name.lower()
        search_list = agents if agents is not None else list(self.agents.values())
        for a in search_list:
            if a.name.lower() == name_lower:
                return a
        return None

    # ============== ACTION EXECUTION (ENHANCED) ==============
    def execute_action(self, agent: Agent, decision: Dict, others: List[Agent]) -> str:
        """Execute an agent's decided action with skill improvements."""
        action = decision.get("action", "REST").upper()
        target = decision.get("target")
        speech = decision.get("speech")
        
        result = ""
        skill_bonus = agent.get_skill_bonus(action)

        # ============== BASIC ACTIONS ==============
        if action == "REST":
            recovery = 3 + (agent.skills.get("survival", 0) // 20)
            agent.energy = min(agent.energy + recovery, 20)
            agent.improve_skill("survival", 1)
            result = f"😴 {agent.name} rested and recovered {recovery} energy (now {agent.energy})"
            agent.add_memory("I took time to rest and recover my strength.")

        elif action == "OBSERVE":
            observations = [o.name for o in others]
            agent.improve_skill("observation", 1)
            if observations:
                # Higher observation skill reveals more info
                if agent.skills.get("observation", 0) > 20:
                    details = [f"{o.name} (bits:{o.bits})" for o in others]
                    result = f"👁 {agent.name} observed: {', '.join(details)}"
                    agent.add_memory(f"I carefully watched {', '.join(observations)} and learned their wealth levels.")
                else:
                    result = f"👁 {agent.name} observed: {', '.join(observations)}"
                    agent.add_memory(f"I watched {', '.join(observations)} at {agent.location}")
            else:
                result = f"👁 {agent.name} observed the quiet surroundings"
                agent.add_memory(f"I observed {agent.location} but no one was around.")

        elif action == "MOVE":
            if target and target in LOCATIONS and target != agent.location:
                cost = MOVEMENT_COSTS.get((agent.location, target), 3)
                if agent.energy >= cost:
                    old_loc = agent.location
                    agent.energy -= cost
                    agent.location = target
                    result = f"🚶 {agent.name} traveled from {old_loc} to {target} (-{cost} energy)"
                    agent.add_memory(f"I traveled from {old_loc} to {target}.")
                else:
                    result = f"😓 {agent.name} is too tired to travel to {target}"
                    agent.add_memory(f"I wanted to go to {target} but was too exhausted.")
            else:
                result = f"❓ {agent.name} tried to move but got confused"

        # ============== SOCIAL ACTIONS ==============
        elif action == "TALK":
            other = self.find_agent_by_name(target, others)
            if other:
                delta = random.randint(-3, 8) + skill_bonus
                if "Generous" in agent.personality_traits: delta += 3
                if "Suspicious" in agent.personality_traits: delta -= 2
                if "Honest" in agent.personality_traits: delta += 2
                
                agent.update_relationship(other.id, delta)
                other.update_relationship(agent.id, delta)
                agent.improve_skill("negotiation", 1)
                
                speech_text = f' "{speech}"' if speech else ""
                direction = "improved" if delta > 0 else "worsened" if delta < 0 else "unchanged"
                result = f"💬 {agent.name} talked with {other.name}{speech_text} (relationship {direction})"
                
                agent.add_memory(f"I had a conversation with {other.name}. It went {'well' if delta > 0 else 'poorly'}.")
                other.add_memory(f"{agent.name} talked to me.{speech_text}")
            else:
                result = f"❓ {agent.name} tried to talk to {target} but couldn't find them"

        elif action == "TRADE":
            other = self.find_agent_by_name(target, others)
            if other:
                success_chance = 40 + agent.get_relationship(other.id) + skill_bonus * 5
                if random.randint(0, 100) < success_chance:
                    amount = random.randint(1, 3) + skill_bonus
                    agent.bits += amount
                    other.bits = max(0, other.bits - amount)
                    agent.update_relationship(other.id, 5)
                    other.update_relationship(agent.id, 5)
                    agent.improve_skill("negotiation", 2)
                    result = f"🤝 {agent.name} traded with {other.name} (+{amount} bits)"
                    agent.add_memory(f"I made a successful trade with {other.name}, gaining {amount} bits.", importance=2)
                    other.add_memory(f"I traded with {agent.name}.")
                else:
                    result = f"🚫 {agent.name}'s trade with {other.name} fell through"
                    agent.add_memory(f"My trade negotiation with {other.name} failed.")
            else:
                result = f"❓ {agent.name} tried to trade with {target} but couldn't find them"

        elif action == "GOSSIP":
            other = self.find_agent_by_name(target, others)
            if other:
                # Share knowledge about a third party
                possible_subjects = [a for a in self.agents.values() 
                                    if a.id != agent.id and a.id != other.id]
                if possible_subjects:
                    subject = random.choice(possible_subjects)
                    agent_opinion = agent.get_relationship(subject.id)
                    
                    # Share info based on trust
                    trust = agent.get_relationship(other.id)
                    if trust > 20:
                        other.learn_gossip(subject.id, agent.id, agent_opinion)
                        # Listener adjusts opinion slightly toward gossiper's opinion
                        shift = (agent_opinion - other.get_relationship(subject.id)) // 4
                        other.update_relationship(subject.id, shift)
                        
                        agent.improve_skill("diplomacy", 1)
                        agent.update_relationship(other.id, 3)
                        
                        result = f"🗣️ {agent.name} gossiped with {other.name} about {subject.name}"
                        agent.add_memory(f"I shared my thoughts about {subject.name} with {other.name}.")
                        other.add_memory(f"{agent.name} told me things about {subject.name}.", importance=2)
                    else:
                        result = f"🤷 {other.name} didn't trust {agent.name}'s gossip"
                else:
                    result = f"🤷 {agent.name} had no one to gossip about"
            else:
                result = f"❓ {agent.name} tried to gossip with {target} but couldn't find them"

        elif action == "STEAL":
            other = self.find_agent_by_name(target, others)
            if other:
                success_chance = 25 + skill_bonus * 5
                if "Cunning" in agent.personality_traits: success_chance += 20
                if "Aggressive" in agent.personality_traits: success_chance += 10
                
                if random.randint(0, 100) < success_chance:
                    stolen = min(other.bits, random.randint(1, 4) + skill_bonus)
                    agent.bits += stolen
                    other.bits -= stolen
                    other.update_relationship(agent.id, -30)
                    agent.improve_skill("stealth", 3)
                    result = f"🦹 {agent.name} stole {stolen} bits from {other.name}!"
                    agent.add_memory(f"I successfully stole {stolen} bits from {other.name}.", importance=2)
                    other.add_memory(f"{agent.name} STOLE from me! I lost {stolen} bits. Never trust them!", importance=3)
                else:
                    agent.update_relationship(other.id, -10)
                    other.update_relationship(agent.id, -25)
                    agent.improve_skill("stealth", 1)
                    result = f"🚨 {agent.name} tried to steal from {other.name} but got CAUGHT!"
                    agent.add_memory(f"I tried to steal from {other.name} but was caught. Embarrassing!")
                    other.add_memory(f"{agent.name} tried to steal from me! I caught them red-handed!", importance=2)
            else:
                result = f"❓ {agent.name} tried to steal but found no target"

        elif action == "ALLY":
            other = self.find_agent_by_name(target, others)
            if other:
                threshold = 20 - skill_bonus * 2
                if agent.get_relationship(other.id) > threshold:
                    agent.update_relationship(other.id, 30)
                    other.update_relationship(agent.id, 30)
                    agent.improve_skill("diplomacy", 2)
                    result = f"🤜🤛 {agent.name} formed an ALLIANCE with {other.name}!"
                    agent.add_memory(f"I formed a powerful alliance with {other.name}!", importance=3)
                    other.add_memory(f"{agent.name} and I are now allies!", importance=3)
                else:
                    result = f"🙅 {other.name} rejected {agent.name}'s alliance proposal"
                    agent.add_memory(f"I proposed an alliance to {other.name} but was rejected.")
            else:
                result = f"❓ {agent.name} tried to ally with {target} but couldn't find them"

        elif action == "BETRAY":
            other = self.find_agent_by_name(target, others)
            if other and agent.get_relationship(other.id) > 0:
                stolen = min(other.bits, random.randint(3, 7) + skill_bonus)
                agent.bits += stolen
                other.bits -= stolen
                agent.influence -= 3
                agent.update_relationship(other.id, -60)
                other.update_relationship(agent.id, -80)
                result = f"🗡️ {agent.name} BETRAYED {other.name}! Stole {stolen} bits!"
                agent.add_memory(f"I betrayed {other.name} and took {stolen} bits. They will hate me forever.", importance=3)
                other.add_memory(f"{agent.name} BETRAYED ME! They stole {stolen} bits! I will never forgive this!", importance=3)
            else:
                result = f"❓ {agent.name} had no one to betray"

        # ============== NEW: GROUP ACTIONS ==============
        elif action == "COUNCIL":
            allies_here = [o for o in others if agent.get_relationship(o.id) > 30]
            if len(allies_here) >= 1:
                influence_gain = 2 + len(allies_here) + skill_bonus
                for ally in allies_here:
                    ally.influence += influence_gain // 2
                agent.influence += influence_gain
                agent.improve_skill("leadership", 2)
                ally_names = ", ".join([a.name for a in allies_here])
                result = f"👥 {agent.name} held a COUNCIL with {ally_names}! (+{influence_gain} influence)"
                agent.add_memory(f"I held a council with my allies. Our influence grows!", importance=2)
            else:
                result = f"❓ {agent.name} tried to form a council but had no allies present"

        elif action == "RAID":
            other = self.find_agent_by_name(target, others)
            allies_here = [o for o in others if agent.get_relationship(o.id) > 30 and o.id != (other.id if other else "")]
            if other and len(allies_here) >= 1:
                # Combined raid is more effective
                total_power = 1 + len(allies_here) + skill_bonus
                success_chance = 30 + (total_power * 15)
                
                if random.randint(0, 100) < success_chance:
                    stolen = min(other.bits, random.randint(3, 8) * total_power)
                    share = stolen // (len(allies_here) + 1)
                    agent.bits += share
                    for ally in allies_here:
                        ally.bits += share
                    other.bits -= stolen
                    other.update_relationship(agent.id, -50)
                    for ally in allies_here:
                        other.update_relationship(ally.id, -40)
                    agent.improve_skill("leadership", 2)
                    result = f"⚔️ {agent.name} led a RAID on {other.name} with allies! Stole {stolen} bits!"
                    agent.add_memory(f"Led a successful raid on {other.name} with my allies!", importance=3)
                else:
                    result = f"🛡️ {other.name} defended against the raid from {agent.name}'s group!"
                    other.add_memory(f"{agent.name} tried to raid me with allies but I held them off!", importance=2)
            else:
                result = f"❓ {agent.name} needs allies present to raid"

        elif action == "EXPEDITION":
            if target and target in LOCATIONS:
                allies_here = [o for o in others if agent.get_relationship(o.id) > 30]
                if allies_here:
                    cost = MOVEMENT_COSTS.get((agent.location, target), 3)
                    # Group travel is cheaper
                    reduced_cost = max(1, cost - len(allies_here))
                    
                    if agent.energy >= reduced_cost:
                        old_loc = agent.location
                        agent.energy -= reduced_cost
                        agent.location = target
                        agent.improve_skill("leadership", 1)
                        
                        travelers = [agent.name]
                        for ally in allies_here:
                            if ally.energy >= reduced_cost:
                                ally.energy -= reduced_cost
                                ally.location = target
                                travelers.append(ally.name)
                        
                        result = f"🧭 EXPEDITION! {', '.join(travelers)} traveled together to {target}!"
                        agent.add_memory(f"Led an expedition to {target} with allies!", importance=2)
                    else:
                        result = f"😓 Not enough energy for expedition to {target}"
                else:
                    result = f"❓ {agent.name} needs allies to form an expedition"
            else:
                result = f"❓ {agent.name} specified invalid expedition destination"

        # ============== NEW: REVIVAL ==============
        elif action == "REVIVE":
            dormant_at_loc = [a for a in self.get_dormant_agents() if a.location == agent.location]
            other = self.find_agent_by_name(target, dormant_at_loc)
            if other:
                # Must have positive relationship and resources
                if agent.get_relationship(other.id) > 40 and agent.bits >= 5 and agent.energy >= 3:
                    agent.bits -= 5
                    agent.energy -= 3
                    other.is_dormant = False
                    other.energy = 5
                    other.bits = 0
                    other.revival_count += 1
                    agent.improve_skill("survival", 2)
                    agent.update_relationship(other.id, 20)
                    other.update_relationship(agent.id, 30)
                    result = f"✨ {agent.name} REVIVED {other.name} from dormancy!"
                    agent.add_memory(f"I brought {other.name} back from dormancy. They owe me!", importance=3)
                    other.add_memory(f"{agent.name} saved me from dormancy! I am forever grateful!", importance=3)
                else:
                    if agent.get_relationship(other.id) <= 40:
                        result = f"❓ {agent.name} doesn't have a strong enough bond with {other.name} to revive them"
                    else:
                        result = f"❓ {agent.name} doesn't have enough resources to revive {other.name} (need 5 bits, 3 energy)"
            else:
                result = f"❓ {agent.name} tried to revive {target} but couldn't find them"

        else:
            result = f"❓ {agent.name} seemed confused and did nothing"
            agent.add_memory("I felt confused and couldn't decide what to do.")

        return result

    def run_tick(self) -> List[str]:
        """Run one tick of the simulation."""
        self.tick_count += 1
        print(f"\n{'='*60}")
        print(f"📍 TICK {self.tick_count}")
        print('='*60)
        
        results = []
        active = self.get_active_agents()
        
        if not active:
            print("  ⚠ No active agents!")
            return results
        
        # NEW: Check for world events
        self.trigger_random_event()
        
        random.shuffle(active)
        
        for agent in active:
            # Check for dormancy
            if agent.energy <= 0:
                agent.is_dormant = True
                result = f"💀 {agent.name} has gone DORMANT (0 energy)"
                results.append(result)
                self.log(result, importance=3)
                continue
            
            # Get perception
            others = [a for a in self.get_agents_at_location(agent.location) 
                      if a.id != agent.id]
            
            print(f"\n  🤖 {agent.name}'s turn (Energy: {agent.energy}, Bits: {agent.bits})")
            print(f"     Location: {agent.location}, Others here: {[o.name for o in others]}")
            
            # Build prompt and get decision
            prompt = self.build_prompt(agent, others)
            decision = self.get_llm_decision(agent, prompt)
            
            print(f"     Decision: {decision.get('action')} → {decision.get('target')}")
            reasoning = decision.get('reasoning', 'N/A')
            print(f"     Reasoning: {reasoning[:60]}..." if len(reasoning) > 60 else f"     Reasoning: {reasoning}")
            
            # Execute action
            result = self.execute_action(agent, decision, others)
            results.append(result)
            self.log(result)
            
            # Energy decay
            agent.energy = max(0, agent.energy - 1)
        
        # Show summary
        print(f"\n📊 TICK {self.tick_count} SUMMARY:")
        print("-" * 50)
        for agent in sorted(self.agents.values(), key=lambda a: a.bits, reverse=True):
            status = "💀" if agent.is_dormant else "✓"
            top_skill = max(agent.skills.items(), key=lambda x: x[1]) if any(agent.skills.values()) else ("none", 0)
            skill_str = f"{top_skill[0][:4]}:{top_skill[1]}" if top_skill[1] > 0 else ""
            print(f"  {status} {agent.name:12} | E:{agent.energy:2} | B:{agent.bits:3} | I:{agent.influence:2} | {skill_str:10} | {agent.location}")
        
        self.save_data()
        return results


# ============== MAIN FUNCTION ==============
def main():
    """Main entry point with enhanced command handling."""
    sim = Simulation()
    
    # Initialize if no agents
    if len(sim.agents) == 0:
        print("\n🌱 No agents found. Creating initial population...")
        sim.initialize_default_agents()
        sim.save_data()
    
    print(f"\n✓ Ecosystem loaded with {len(sim.agents)} agents")
    print(f"✓ Current tick: {sim.tick_count}")
    
    print("\n" + "="*60)
    print("COMMANDS:")
    print("  [Enter]  - Run one tick")
    print("  auto     - Auto-run ticks (Ctrl+C to stop)")
    print("  status   - Show all agents with details")
    print("  stats    - Show ecosystem statistics dashboard")
    print("  reset    - Reset simulation")
    print("  quit     - Exit")
    print("="*60)
    
    while True:
        try:
            cmd = input("\n> ").strip().lower()
            
            if cmd == "" or cmd == "tick":
                sim.run_tick()
                
            elif cmd == "auto":
                print("🔄 Auto-running... Press Ctrl+C to stop")
                try:
                    while True:
                        sim.run_tick()
                        time.sleep(5)
                except KeyboardInterrupt:
                    print("\n⏹ Auto-run stopped")
                    
            elif cmd == "status":
                print("\n📋 AGENT STATUS:")
                print("-" * 70)
                for agent in sorted(sim.agents.values(), key=lambda a: a.bits, reverse=True):
                    status = "💀 DORMANT" if agent.is_dormant else "✓ Active"
                    print(f"\n  {agent.name} ({agent.id}) - {status}")
                    print(f"    Drive: {agent.core_drive}")
                    print(f"    Traits: {', '.join(agent.personality_traits)}")
                    print(f"    Location: {agent.location}")
                    print(f"    Energy: {agent.energy} | Bits: {agent.bits} | Influence: {agent.influence}")
                    
                    # Show skills
                    skills_str = ", ".join([f"{k}:{v}" for k, v in agent.skills.items() if v > 0])
                    if skills_str:
                        print(f"    Skills: {skills_str}")
                    
                    # Show relationships
                    if agent.relationships:
                        rels = []
                        for other_id, score in sorted(agent.relationships.items(), key=lambda x: x[1], reverse=True)[:3]:
                            if other_id in sim.agents:
                                other_name = sim.agents[other_id].name
                                emoji = "💚" if score > 30 else "💔" if score < -30 else "😐"
                                rels.append(f"{emoji}{other_name}:{score}")
                        if rels:
                            print(f"    Relationships: {', '.join(rels)}")
                    
                    # Show revival count
                    if agent.revival_count > 0:
                        print(f"    Revivals: {agent.revival_count}")

            elif cmd == "stats":
                sim.show_statistics()
                    
            elif cmd == "reset":
                confirm = input("⚠ Delete all data and restart? (yes/no): ")
                if confirm.lower() == "yes":
                    if os.path.exists(DATA_FILE):
                        os.remove(DATA_FILE)
                    sim = Simulation()
                    sim.initialize_default_agents()
                    sim.save_data()
                    print("✓ Simulation reset!")
                    
            elif cmd in ["quit", "exit", "q"]:
                print("👋 Goodbye, Great Observer!")
                break
                
            else:
                print("Unknown command. Try: tick, auto, status, stats, reset, quit")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()