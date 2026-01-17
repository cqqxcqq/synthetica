"""
Streamlit Cloud version of Synthetica Ecosystem
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pandas as pd
import json
import os
import random
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from groq import Groq
import chromadb
from chromadb.config import Settings

# ============== CONFIGURATION ==============
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

MODEL_NAME = "llama3-8b-8192"
LOCATIONS = ["The Archive", "The Market", "The Void", "The Nexus", "The Garden"]

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

# ============== AGENT CLASS ==============
class Agent:
    def __init__(self, name: str, core_drive: str, personality_traits: List[str],
                 agent_id: str = None, location: str = None, energy: int = 10,
                 bits: int = 5, influence: int = 0, is_dormant: bool = False,
                 relationships: Dict = None):
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

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "name": self.name, "core_drive": self.core_drive,
            "personality_traits": self.personality_traits, "location": self.location,
            "energy": self.energy, "bits": self.bits, "influence": self.influence,
            "is_dormant": self.is_dormant, "relationships": self.relationships
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Agent':
        return cls(
            agent_id=data["id"], name=data["name"], core_drive=data["core_drive"],
            personality_traits=data["personality_traits"], location=data["location"],
            energy=data["energy"], bits=data["bits"], influence=data["influence"],
            is_dormant=data["is_dormant"], relationships=data.get("relationships", {})
        )

    def update_relationship(self, other_id: str, delta: int):
        current = self.relationships.get(other_id, 0)
        self.relationships[other_id] = max(-100, min(100, current + delta))

    def get_relationship(self, other_id: str) -> int:
        return self.relationships.get(other_id, 0)


# ============== SIMULATION CLASS ==============
class SyntheticaSimulation:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
        self.agents: Dict[str, Agent] = {}
        self.tick_count = 0
        self.logs = []
        self.world_resources = {"chaos_level": 0, "harmony_level": 50}
        
        # In-memory ChromaDB for cloud
        self.chroma_client = chromadb.Client()
        self.memories = self.chroma_client.get_or_create_collection("memories")

    def create_agent(self, name: str = None, core_drive: str = None, 
                     traits: List[str] = None) -> Agent:
        if name is None:
            name = f"Agent_{len(self.agents) + 1:03d}"
        if core_drive is None:
            core_drive = random.choice(CORE_DRIVES)
        if traits is None:
            traits = random.sample(PERSONALITY_TRAITS, random.randint(2, 4))
        
        agent = Agent(name=name, core_drive=core_drive, personality_traits=traits)
        self.agents[agent.id] = agent
        return agent

    def initialize_default_agents(self):
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

    def get_agents_at_location(self, location: str) -> List[Agent]:
        return [a for a in self.agents.values() if a.location == location and not a.is_dormant]

    def get_active_agents(self) -> List[Agent]:
        return [a for a in self.agents.values() if not a.is_dormant]

    def store_memory(self, agent_id: str, memory: str, related: List[str] = None):
        mem_id = f"{agent_id}_{datetime.now().timestamp()}"
        self.memories.add(
            documents=[memory],
            metadatas=[{"agent_id": agent_id, "related": ",".join(related or [])}],
            ids=[mem_id]
        )

    def recall_memories(self, agent_id: str, query: str, n: int = 3) -> List[str]:
        try:
            results = self.memories.query(
                query_texts=[query], n_results=n,
                where={"agent_id": agent_id}
            )
            return results['documents'][0] if results['documents'] else []
        except:
            return []

    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        self.logs = self.logs[-100:]  # Keep last 100

    def build_prompt(self, agent: Agent, others: List[Agent], memories: List[str]) -> str:
        others_text = "\n".join([
            f"  - {o.name} (relationship: {agent.get_relationship(o.id)})"
            for o in others
        ]) or "  No one else here"
        
        memories_text = "\n".join([f"  - {m}" for m in memories]) or "  No relevant memories"
        
        return f"""You are {agent.name}, an autonomous agent.

IDENTITY:
- Core Drive: {agent.core_drive}
- Traits: {', '.join(agent.personality_traits)}

STATUS:
- Location: {agent.location}
- Energy: {agent.energy} | Bits: {agent.bits} | Influence: {agent.influence}

OTHERS PRESENT:
{others_text}

MEMORIES:
{memories_text}

ACTIONS: TALK, TRADE, MOVE, REST, STEAL, ALLY, BETRAY, OBSERVE

Respond in JSON:
{{"action": "ACTION", "target": "name or location or null", "reasoning": "why", "dialogue": "what you say or null"}}"""

    def get_llm_decision(self, agent: Agent, prompt: str) -> Dict:
        if not self.client:
            return {"action": "REST", "target": None, "reasoning": "No API key", "dialogue": None}
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an autonomous agent. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=300
            )
            text = response.choices[0].message.content.strip()
            
            # Extract JSON
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            return json.loads(text)
        except Exception as e:
            return {"action": "OBSERVE", "target": None, "reasoning": str(e), "dialogue": None}

    def execute_action(self, agent: Agent, decision: Dict, others: List[Agent]) -> str:
        action = decision.get("action", "REST").upper()
        target = decision.get("target")
        dialogue = decision.get("dialogue")
        
        result = ""
        
        if action == "REST":
            agent.energy = min(agent.energy + 2, 20)
            result = f"{agent.name} rested (+2 energy)"
            
        elif action == "OBSERVE":
            result = f"{agent.name} observed the surroundings"
            
        elif action == "MOVE" and target in LOCATIONS:
            cost = MOVEMENT_COSTS.get((agent.location, target), 3)
            if agent.energy >= cost:
                old = agent.location
                agent.energy -= cost
                agent.location = target
                result = f"{agent.name} moved from {old} to {target}"
            else:
                result = f"{agent.name} too tired to move"
                
        elif action == "TALK" and target:
            other = next((o for o in others if o.name.lower() == target.lower()), None)
            if other:
                delta = random.randint(-5, 10)
                agent.update_relationship(other.id, delta)
                other.update_relationship(agent.id, delta)
                result = f"{agent.name} talked with {target}"
                if dialogue:
                    result += f': "{dialogue}"'
            else:
                result = f"{agent.name} couldn't find {target}"
                
        elif action == "TRADE" and target:
            other = next((o for o in others if o.name.lower() == target.lower()), None)
            if other and random.random() < 0.6:
                amount = random.randint(1, 3)
                agent.bits += amount
                other.bits -= amount
                agent.update_relationship(other.id, 5)
                other.update_relationship(agent.id, 5)
                result = f"{agent.name} traded with {target} (+{amount} bits)"
            else:
                result = f"{agent.name}'s trade with {target} failed"
                
        elif action == "STEAL" and target:
            other = next((o for o in others if o.name.lower() == target.lower()), None)
            if other and random.random() < 0.3:
                stolen = min(other.bits, random.randint(1, 3))
                agent.bits += stolen
                other.bits -= stolen
                other.update_relationship(agent.id, -25)
                result = f"{agent.name} stole {stolen} bits from {target}!"
                self.store_memory(other.id, f"{agent.name} stole from me!", [agent.id])
            else:
                agent.update_relationship(other.id if other else "", -10)
                result = f"{agent.name} failed to steal from {target}"
                
        elif action == "ALLY" and target:
            other = next((o for o in others if o.name.lower() == target.lower()), None)
            if other and agent.get_relationship(other.id) > 20:
                agent.update_relationship(other.id, 25)
                other.update_relationship(agent.id, 25)
                result = f"{agent.name} formed alliance with {target}!"
                self.store_memory(agent.id, f"Allied with {target}", [other.id])
                self.store_memory(other.id, f"Allied with {agent.name}", [agent.id])
            else:
                result = f"{agent.name}'s alliance proposal rejected by {target}"
                
        elif action == "BETRAY" and target:
            other = next((o for o in others if o.name.lower() == target.lower()), None)
            if other and agent.get_relationship(other.id) > 0:
                stolen = min(other.bits, random.randint(2, 5))
                agent.bits += stolen
                other.bits -= stolen
                agent.update_relationship(other.id, -50)
                other.update_relationship(agent.id, -60)
                result = f"{agent.name} BETRAYED {target}! Stole {stolen} bits!"
                self.store_memory(other.id, f"{agent.name} betrayed me!", [agent.id])
            else:
                result = f"{agent.name} had no one to betray"
        else:
            result = f"{agent.name} did nothing"
        
        self.store_memory(agent.id, result, [])
        return result

    def run_tick(self) -> List[str]:
        self.tick_count += 1
        results = []
        self.log(f"=== TICK {self.tick_count} ===")
        
        active = self.get_active_agents()
        random.shuffle(active)
        
        for agent in active:
            if agent.energy <= 0:
                agent.is_dormant = True
                results.append(f"💀 {agent.name} went dormant!")
                continue
            
            others = [a for a in self.get_agents_at_location(agent.location) if a.id != agent.id]
            memories = self.recall_memories(agent.id, "recent events")
            prompt = self.build_prompt(agent, others, memories)
            decision = self.get_llm_decision(agent, prompt)
            result = self.execute_action(agent, decision, others)
            
            agent.energy -= 1
            results.append(result)
            self.log(result)
        
        return results

    def get_leaderboard(self) -> List[Dict]:
        board = []
        for agent in self.agents.values():
            power = agent.bits + agent.energy + (agent.influence * 2)
            board.append({
                "name": agent.name, "bits": agent.bits, "energy": agent.energy,
                "influence": agent.influence, "power": power, "dormant": agent.is_dormant
            })
        return sorted(board, key=lambda x: x["power"], reverse=True)

    def god_action(self, action: str, params: Dict) -> str:
        if action == "chaos":
            for a in self.agents.values():
                for b in self.agents.values():
                    if a.id != b.id:
                        a.relationships[b.id] = random.randint(-50, 50)
            return "🌪️ CHAOS! All relationships scrambled!"
        elif action == "revive":
            for a in self.agents.values():
                if a.is_dormant:
                    a.is_dormant = False
                    a.energy = 10
            return "✨ All agents revived!"
        elif action == "gift":
            for a in self.agents.values():
                a.bits += 5
                a.energy = min(a.energy + 5, 20)
            return "🎁 Gifted resources to all agents!"
        return "Unknown action"


# ============== STREAMLIT UI ==============
st.set_page_config(page_title="Synthetica", page_icon="🌐", layout="wide")

# Initialize simulation in session state
if 'sim' not in st.session_state:
    st.session_state.sim = SyntheticaSimulation()
    
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

sim = st.session_state.sim

# Header
st.title("🌐 The Synthetica Ecosystem")
st.caption("A self-operating agent-based social simulation powered by Groq LLM")

# Check API key
if not GROQ_API_KEY:
    st.error("⚠️ No GROQ_API_KEY found! Add it to Streamlit secrets or environment.")
    st.code("GROQ_API_KEY = 'gsk_your_key_here'", language="toml")
    st.stop()

# Initialize agents
if not st.session_state.initialized:
    if st.button("🚀 Initialize Ecosystem", type="primary", use_container_width=True):
        sim.initialize_default_agents()
        st.session_state.initialized = True
        st.rerun()
    st.info("Click above to create the initial agents and start the simulation!")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("🎮 Control Panel")
    
    if st.button("▶️ Run Tick", use_container_width=True, type="primary"):
        with st.spinner("Running tick..."):
            results = sim.run_tick()
            st.session_state.last_results = results
        st.rerun()
    
    st.markdown("---")
    st.subheader("👁️ God Console")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌪️ Chaos"):
            result = sim.god_action("chaos", {})
            st.toast(result)
    with col2:
        if st.button("✨ Revive"):
            result = sim.god_action("revive", {})
            st.toast(result)
    
    if st.button("🎁 Gift All", use_container_width=True):
        result = sim.god_action("gift", {})
        st.toast(result)
    
    st.markdown("---")
    st.subheader("➕ Add Agent")
    new_name = st.text_input("Name")
    if st.button("Create", use_container_width=True):
        if new_name:
            sim.create_agent(name=new_name)
            st.toast(f"Created {new_name}!")
            st.rerun()

# Main content
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tick", sim.tick_count)
col2.metric("Active", len(sim.get_active_agents()))
col3.metric("Total", len(sim.agents))
col4.metric("Dormant", len([a for a in sim.agents.values() if a.is_dormant]))

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🕸️ Relationships", "👥 Agents", "📜 Logs"])

with tab1:
    # Leaderboard
    st.subheader("🏆 Power Leaderboard")
    board = sim.get_leaderboard()
    if board:
        df = pd.DataFrame(board)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        fig = px.bar(df, x='name', y='power', color='dormant',
                     color_discrete_map={True: 'gray', False: 'steelblue'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent actions
    st.subheader("⚡ Recent Actions")
    if hasattr(st.session_state, 'last_results'):
        for r in st.session_state.last_results:
            st.info(r)
    else:
        st.caption("Run a tick to see actions!")

with tab2:
    st.subheader("🕸️ Relationship Network")
    
    G = nx.DiGraph()
    for agent in sim.agents.values():
        G.add_node(agent.id, name=agent.name, dormant=agent.is_dormant)
        for other_id, score in agent.relationships.items():
            if score != 0:
                G.add_edge(agent.id, other_id, weight=score)
    
    if len(G.nodes) > 0:
        pos = nx.spring_layout(G, k=2)
        
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            color = "green" if edge[2]['weight'] > 0 else "red"
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines', line=dict(width=abs(edge[2]['weight'])/20, color=color),
                hoverinfo='none'
            ))
        
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_text = [G.nodes[n]['name'] for n in G.nodes()]
        node_colors = ['gray' if G.nodes[n]['dormant'] else 'steelblue' for n in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            text=node_text, textposition="top center",
            marker=dict(size=20, color=node_colors)
        )
        
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(showlegend=False, height=500,
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("🟢 Green = Positive | 🔴 Red = Negative | Gray = Dormant")

with tab3:
    st.subheader("👥 Agent Details")
    for agent in sim.agents.values():
        icon = "💀" if agent.is_dormant else "🤖"
        with st.expander(f"{icon} {agent.name}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Drive:** {agent.core_drive}")
                st.markdown(f"**Traits:** {', '.join(agent.personality_traits)}")
                st.markdown(f"**Location:** {agent.location}")
            with c2:
                st.metric("Energy", agent.energy)
                st.metric("Bits", agent.bits)
                st.metric("Influence", agent.influence)

with tab4:
    st.subheader("📜 Event Log")
    if sim.logs:
        log_text = "\n".join(reversed(sim.logs))
        st.text_area("Logs", log_text, height=400)
    else:
        st.caption("No events yet. Run some ticks!")

# Footer
st.markdown("---")
st.caption("Built with Groq LLM + Streamlit | The Great Observer watches...")
