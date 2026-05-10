#!/usr/bin/env python3
"""Generate diverse insurance broker outbound call transcripts with integrated system prompts.

Merges dialogue generation and system prompt generation into a single step.
Uses an AttrPrompt-inspired seed matrix with 7-dimensional Latin Hypercube
Sampling, including weighted axes for call outcomes (80/20 success/failure)
and context injections (50% with / 50% without).

Each output record contains:
  - text_prompt: structured pre-call brief (ROLE/CLIENT/KNOWN/UNKNOWN/GOAL/STYLE)
  - dialogue: natural phone call transcript with backchanneling and interruptions
  - context_injections: list of mid-call injection points (for puppeteer training)

Dependencies:
    pip install anthropic numpy

Usage:
    python generate_dialogues_sync.py                          # 200 dialogues
    python generate_dialogues_sync.py -n 500 -o batch2.jsonl   # custom
    python generate_dialogues_sync.py --dry-run -n 3           # preview prompts
"""

import argparse
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np

import anthropic

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed Axis 1: Broker Personas (60)
# ---------------------------------------------------------------------------
_BROKER_NAMES = [
    "Margaret Chen", "Robert Blackwell", "Sarah Okafor", "James Whitfield",
    "Patricia Hernandez", "William Tanaka", "Linda Johansson", "Michael Dubois",
    "Jennifer Patel", "David Kowalski", "Maria Santos", "Richard O'Brien",
    "Susan Kim", "Charles Fitzgerald", "Karen Nakamura", "Thomas Abadi",
    "Nancy Petrov", "Daniel Moreau", "Lisa Ramirez", "Paul Bergström",
    "Betty Washington", "Mark Antonelli", "Sandra Osei", "Steven Yamamoto",
    "Dorothy Fischer", "Andrew Castillo", "Ashley Krishnamurthy", "Joshua Lindqvist",
    "Kimberly Okonkwo", "Brian Arsenault", "Emily Zhao", "Kevin McAllister",
    "Amanda Becker", "Jason Nkomo", "Melissa Chang", "Ryan Gupta",
    "Stephanie Larsson", "Gary Diallo", "Rebecca Novak", "Timothy Hassan",
    "Laura Svensson", "Jeffrey Abubakar", "Cynthia Morales", "Frank Taniguchi",
    "Kathleen Popov", "Scott Delacroix", "Angela Watanabe", "Raymond Oduya",
    "Shirley Magnusson", "Dennis Chakraborty", "Michelle Fournier", "Larry Suzuki",
    "Carol Mensah", "Gregory Volkov", "Diane Takahashi", "Jerry Andersen",
    "Heather Mbeki", "Henry Iglesias", "Teresa Kovalenko", "Arthur Nakagawa",
]

_EXPERIENCE_LEVELS = [
    ("junior", "1-3 years in insurance, still learning nuances"),
    ("developing", "4-7 years, solid foundation but defers on complex cases"),
    ("experienced", "8-15 years, handles most situations confidently"),
    ("senior", "16-25 years, deep expertise, mentors others"),
    ("veteran", "25+ years, industry authority, has seen every scenario"),
]

_SPECIALTIES = [
    "personal lines generalist", "commercial property & casualty",
    "professional liability (E&O / D&O)", "workers compensation",
    "commercial auto & fleet", "construction & surety bonds",
    "cyber liability & tech E&O", "life & health benefits",
    "marine & cargo", "environmental liability",
    "excess & surplus lines", "high-net-worth personal lines",
]

_COMM_STYLES = [
    "methodical and detail-oriented — walks through every clause",
    "warm and relationship-focused — builds rapport before business",
    "direct and efficient — gets straight to the point",
    "educational — explains insurance concepts patiently",
    "consultative — asks probing questions to uncover needs",
]

_REGIONAL_PATTERNS = [
    "Northeast US — faster pace, occasional local idioms",
    "Southern US — measured pace, polite and conversational",
    "Midwest US — straightforward, friendly, no-nonsense",
    "West Coast US — casual tone, tech-savvy vocabulary",
    "Canadian — courteous, measured, bilingual references occasionally",
    "UK-influenced — formal phrasing, Lloyds/London market references",
]


def _build_broker_personas() -> list[dict]:
    """Build 60 broker personas by cycling through attribute arrays."""
    personas = []
    for i in range(60):
        exp_level, exp_desc = _EXPERIENCE_LEVELS[i % len(_EXPERIENCE_LEVELS)]
        personas.append({
            "name": _BROKER_NAMES[i],
            "experience_level": exp_level,
            "experience_description": exp_desc,
            "specialty": _SPECIALTIES[i % len(_SPECIALTIES)],
            "communication_style": _COMM_STYLES[i % len(_COMM_STYLES)],
            "regional_speech": _REGIONAL_PATTERNS[i % len(_REGIONAL_PATTERNS)],
        })
    return personas


# ---------------------------------------------------------------------------
# Seed Axis 2: Client Personas (80)
# ---------------------------------------------------------------------------
_CLIENT_NAMES = [
    "James Whitfield", "Priya Mehta", "Carlos Gutierrez", "Anna Kowalczyk",
    "Darnell Washington", "Yuki Tanaka", "Fatima Al-Rashid", "Liam O'Sullivan",
    "Chen Wei", "Isabella Rossi", "Kwame Asante", "Sophie Bergeron",
    "Ravi Krishnan", "Elena Volkov", "Marcus Thompson", "Hana Suzuki",
    "Ahmed Hassan", "Natalie Johansson", "Diego Morales", "Aisha Diallo",
    "Viktor Petrov", "Grace Okonkwo", "Tomas Novak", "Julia Santos",
    "Hiroshi Nakamura", "Rebecca Andersen", "Omar Farouk", "Ingrid Svensson",
    "Luis Castillo", "Chioma Eze", "Mikhail Sorokin", "Astrid Lindgren",
    "Raj Patel", "Camille Dubois", "Kenji Watanabe", "Amara Mensah",
    "Stefan Fischer", "Mei-Ling Chang", "Ibrahim Nkomo", "Eva Magnusson",
    "Andre Williams", "Sakura Yamamoto", "Olga Kovalenko", "Pierre Delacroix",
    "Nadia Chakraborty", "Erik Larsson", "Folake Adeyemi", "Hans Becker",
    "Sunita Gupta", "Brendan McAllister", "Yuko Takahashi", "Samuel Osei",
    "Katarina Popov", "Jin-Ho Park", "Zainab Abubakar", "Lars Eriksson",
    "Daniela Hernandez", "Takeshi Taniguchi", "Nneka Oduya", "Anton Volkov",
    "Marie Fournier", "Deepak Sharma", "Freya Andersen", "Juan Iglesias",
    "Blessing Mbeki", "Nils Bergström", "Rosa Antonelli", "Haruto Suzuki",
    "Celine Arsenault", "Vikram Singh", "Elsa Lindqvist", "Kofi Mensah",
    "Petra Novak", "Kenichi Nakagawa", "Adanna Okafor", "François Moreau",
    "Suki Kim", "Lars Magnusson", "Ifeoma Diallo", "Gustav Svensson",
]

_COMPANY_SIZES = [
    ("sole proprietor", "1 employee, home-based or mobile"),
    ("micro business", "2-5 employees, single location"),
    ("small business", "6-25 employees, 1-2 locations"),
    ("growing SMB", "26-100 employees, regional presence"),
    ("mid-market", "101-500 employees, multi-state operations"),
    ("upper mid-market", "501-2000 employees, national scope"),
    ("large enterprise", "2000-10000 employees, multiple divisions"),
    ("individual / personal lines", "consumer seeking personal insurance"),
]

_CLIENT_ROLES = [
    "business owner / CEO", "risk manager", "CFO / controller",
    "office manager / admin", "operations director", "HR manager",
    "facilities manager", "fleet manager", "project manager",
    "homeowner / individual consumer",
]

_SOPHISTICATION_LEVELS = [
    ("novice", "minimal insurance knowledge, needs everything explained"),
    ("basic", "understands fundamentals but gets lost in details"),
    ("intermediate", "comfortable with common terms, asks smart questions"),
    ("expert", "deep insurance knowledge, challenges broker on coverage nuances"),
]

_RELATIONSHIP_TENURES = [
    ("prospect", "first interaction, no prior relationship"),
    ("new client", "0-1 year, still building trust"),
    ("established", "2-5 years, comfortable working relationship"),
    ("long-term", "6-15 years, strong loyalty and mutual respect"),
    ("legacy", "15+ years, multi-generational or deeply embedded"),
]


def _build_client_personas() -> list[dict]:
    """Build 80 client personas."""
    personas = []
    for i in range(80):
        soph, soph_desc = _SOPHISTICATION_LEVELS[i % len(_SOPHISTICATION_LEVELS)]
        tenure, tenure_desc = _RELATIONSHIP_TENURES[i % len(_RELATIONSHIP_TENURES)]
        size, size_desc = _COMPANY_SIZES[i % len(_COMPANY_SIZES)]
        personas.append({
            "name": _CLIENT_NAMES[i],
            "company_size": size,
            "company_description": size_desc,
            "role": _CLIENT_ROLES[i % len(_CLIENT_ROLES)],
            "insurance_sophistication": soph,
            "sophistication_description": soph_desc,
            "relationship_tenure": tenure,
            "tenure_description": tenure_desc,
        })
    return personas


# ---------------------------------------------------------------------------
# Seed Axis 3: Broker Outbound Goals (32)
# ---------------------------------------------------------------------------
BROKER_GOALS = [
    # Renewal & Retention (10)
    "Renewal outreach — present renewal terms and address premium increase",
    "Retention save — client is shopping competitors, broker must retain the account",
    "Win-back — re-engage lapsed client with improved terms or new options",
    "Annual coverage review — confirm adequacy and identify gaps since last review",
    "Rate increase notification — deliver premium increase with alternative options",
    "Policy restructuring — propose reorganizing coverage for better protection or savings",
    "Expiring endorsement — rider or endorsement expiring, needs renewal or replacement",
    "Exposure change follow-up — client's operations changed, coverage needs adjustment",
    "Multi-year commitment pitch — offer rate lock or premium stability for longer term",
    "Carrier switch recommendation — found better carrier fit, present the case to switch",
    # Cross-sell & Growth (8)
    "Coverage gap fill — identified uninsured or underinsured exposure, pitch new line",
    "Limits increase — business growth or contract requirements warrant higher limits",
    "Policy consolidation — combine multiple standalone policies into a package for savings",
    "Emerging risk pitch — recommend cyber, employment practices, or other modern coverage",
    "Umbrella recommendation — underlying limits warrant an excess or umbrella layer",
    "Key person coverage — propose life or disability coverage for business continuity",
    "Commercial auto addition — client expanded fleet or vehicles, needs auto coverage",
    "Professional liability addition — service operations expanding, needs E&O coverage",
    # Proactive Service (8)
    "Audit data collection — need payroll, revenue, or fleet data before audit deadline",
    "Claim status update — inform client about open claim progress and next steps",
    "Loss control follow-up — carrier recommendations need implementation to avoid surcharge",
    "Certificate request — need additional info from client to issue COI for their contract",
    "Premium payment resolution — payment late or returned, resolve before cancellation notice",
    "Policy documentation — missing signatures or forms needed to complete policy file",
    "Compliance deadline alert — regulatory or contractual insurance requirement approaching",
    "Safety program enrollment — recommend carrier loss prevention or risk management program",
    # New Business & Closing (6)
    "Quote follow-up — proposal sent last week, calling to review terms and close",
    "New client onboarding — walk through bound coverage details and set service expectations",
    "Application completion — need missing information to submit to underwriting",
    "Bind authorization — coverage approved, need verbal or written confirmation to bind",
    "Competitive presentation — present quote against client's incumbent broker or carrier",
    "Referral follow-up — mutual contact introduced broker, first substantive outreach call",
]

# ---------------------------------------------------------------------------
# Seed Axis 4: Emotional Tenors (7)
# ---------------------------------------------------------------------------
EMOTIONAL_TENORS = [
    "routine — standard business call, calm and professional",
    "delicate — delivering unwelcome news or navigating a sensitive topic",
    "upbeat — positive energy, good rapport, favorable news to share",
    "urgent — deadline pressure, time-sensitive action required from client",
    "consultative — collaborative problem-solving, weighing options together",
    "tense — strained relationship, prior service issue, or client frustration",
    "persuasive — broker overcoming client reluctance or objections",
]

# ---------------------------------------------------------------------------
# Seed Axis 5: Domain Vocabulary Clusters (14)
# ---------------------------------------------------------------------------
DOMAIN_VOCAB_CLUSTERS = [
    "personal auto — liability limits, collision/comprehensive, UM/UIM, SR-22, telematics, deductibles, rental reimbursement",
    "homeowners — dwelling coverage, personal property, loss of use, replacement cost vs ACV, ordinance & law, scheduled articles",
    "commercial property — building valuation, BPP, business income, coinsurance, agreed value, vacancy clause, equipment breakdown",
    "general liability — occurrence vs claims-made, completed operations, products liability, advertising injury, contractual liability",
    "workers compensation — experience modification rate, NCCI class codes, return-to-work programs, audit, monopolistic states",
    "commercial auto — hired & non-owned, fleet scheduling, MCS-90, cargo, garagekeepers, symbol system",
    "professional liability — claims-made trigger, prior acts, tail coverage, defense costs inside/outside limits, consent to settle",
    "cyber liability — first-party/third-party, breach response, business interruption, social engineering, ransomware, PCI DSS",
    "life & health — term vs whole life, group health, short/long-term disability, key person, buy-sell funding, COBRA",
    "excess & umbrella — following form, self-insured retention, drop-down coverage, scheduled vs unscheduled underlying",
    "construction — builders risk, wrap-up (OCIP/CCIP), subcontractor default, performance bonds, completed operations tail",
    "environmental — pollution legal liability, remediation cost cap, contractors pollution, mold, Superfund, Phase I/II ESA",
    "marine — hull & machinery, protection & indemnity, cargo (warehouse-to-warehouse), general average, salvage, maritime law",
    "business interruption — period of restoration, extended period, contingent BI, civil authority, service interruption, extra expense",
]

# ---------------------------------------------------------------------------
# Seed Axis 6: Complexity Levels (4)
# ---------------------------------------------------------------------------
COMPLEXITY_LEVELS = [
    ("simple", "Single topic, straightforward exchange, resolved or clearly deferred within the call"),
    ("moderate", "Two related topics, some back-and-forth, minor complications"),
    ("complex", "Multiple interrelated coverage issues, requires research or follow-up commitments"),
    ("escalation", "Issue requires underwriter, supervisor, or specialist involvement — broker may need to loop someone in"),
]

# ---------------------------------------------------------------------------
# Seed Axis 7: Call Outcomes — weighted 80/20 success/failure
# ---------------------------------------------------------------------------
OUTCOMES_SUCCESS = [
    ("full_success", "Broker achieves all call objectives within the call"),
    ("partial_success", "Primary goal achieved but secondary items deferred to follow-up"),
    ("deferred_success", "Client agrees in principle, commits to follow-up call or action"),
    ("conditional_success", "Client agrees contingent on additional info, spouse/partner approval, or board decision"),
]

OUTCOMES_FAILURE = [
    ("soft_failure", "Client declines politely or defers indefinitely — relationship preserved"),
    ("hard_failure", "Client pushes back firmly — unresolved tension, possible churn risk"),
]


def map_outcome(lhs_val: float) -> dict:
    """Map a [0, 1) LHS value to a call outcome with 80/20 success/failure split.

    [0.0, 0.8) -> 4 success variants (20% each of total)
    [0.8, 1.0) -> 2 failure variants (10% each of total)
    """
    if lhs_val < 0.8:
        idx = min(int((lhs_val / 0.8) * len(OUTCOMES_SUCCESS)), len(OUTCOMES_SUCCESS) - 1)
        label, desc = OUTCOMES_SUCCESS[idx]
    else:
        idx = min(int(((lhs_val - 0.8) / 0.2) * len(OUTCOMES_FAILURE)), len(OUTCOMES_FAILURE) - 1)
        label, desc = OUTCOMES_FAILURE[idx]
    return {"label": label, "description": desc}


def map_injection_count(lhs_val: float) -> int:
    """Map a [0, 1) LHS value to an injection count with ~50/50 split.

    [0.0, 0.5)  -> 3 injections (50%)
    [0.5, 0.75) -> 0 injections  (25%)
    [0.75, 0.9) -> 6 injections (15%)
    [0.9, 1.0)  -> 9 injections (10%)
    """
    if lhs_val < 0.5:
        return 3
    elif lhs_val < 0.75:
        return 0
    elif lhs_val < 0.9:
        return 6
    else:
        return 9


# ---------------------------------------------------------------------------
# Latin Hypercube Sampling
# ---------------------------------------------------------------------------

def latin_hypercube_sample(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    """Pure-NumPy Latin Hypercube Sampling.

    Returns an (n_samples, n_dims) array of floats in [0, 1).
    Each dimension is divided into n_samples equal strata, and exactly
    one sample is drawn from each stratum.
    """
    result = np.empty((n_samples, n_dims))
    for dim in range(n_dims):
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            low = perm[i] / n_samples
            high = (perm[i] + 1) / n_samples
            result[i, dim] = rng.uniform(low, high)
    return result


def map_samples_to_seeds(
    lhs_matrix: np.ndarray,
    brokers: list[dict],
    clients: list[dict],
    goals: list[str],
    tenors: list[str],
    vocabs: list[str],
    complexities: list[tuple[str, str]],
) -> list[dict]:
    """Map 8-dim LHS samples to concrete seed combinations.

    Dims 0-5: standard uniform mapping to axis.
    Dim 6: weighted outcome mapping [0, 0.8) -> success, [0.8, 1.0) -> failure.
    Dim 7: weighted injection count [0, 0.5) -> 0, [0.5, 0.75) -> 1, etc.
    """
    seeds = []
    axis_sizes = [
        len(brokers), len(clients), len(goals),
        len(tenors), len(vocabs), len(complexities),
    ]
    for row in lhs_matrix:
        indices = [int(row[d] * axis_sizes[d]) for d in range(6)]
        indices = [min(idx, sz - 1) for idx, sz in zip(indices, axis_sizes)]

        outcome = map_outcome(row[6])
        injection_count = map_injection_count(row[7])
        comp_label, comp_desc = complexities[indices[5]]
        seeds.append({
            "broker": brokers[indices[0]],
            "client": clients[indices[1]],
            "goal": goals[indices[2]],
            "tenor": tenors[indices[3]],
            "domain_vocab": vocabs[indices[4]],
            "complexity": comp_label,
            "complexity_description": comp_desc,
            "outcome": outcome,
            "injection_count": injection_count,
        })
    return seeds


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert scriptwriter generating realistic insurance broker OUTBOUND \
phone call transcripts. The broker is calling the client with a specific goal.

You will produce two outputs:
1. A structured pre-call brief (the broker's system prompt before dialing)
2. A complete phone call transcript

=== PRE-CALL BRIEF ===

Write a structured brief in this exact format (UNDER 128 WORDS TOTAL):

ROLE: [full name], [agency name you invent], [specialty area]
CLIENT: [full name], [role], [company name you invent]
RELATIONSHIP: [tenure and rapport description]
KNOWN: [5-8 specific pre-call facts from the broker's CRM/files — policy \
details with real dollar amounts, coverage types, account dates, prior \
interactions. Be concrete with numbers and names.]
UNKNOWN: [2-4 items the broker anticipates needing but doesn't have ready — \
specific quotes, underwriter decisions, competitor details. This is NOT \
exhaustive — the client will also ask things the broker never anticipated.]
GOAL: [the broker's specific call objective]
STYLE: [communication style and regional speech pattern from broker profile]

=== TRANSCRIPT RULES ===

This must sound like a REAL phone call. The following patterns are MANDATORY.

BACKCHANNELING — at MINIMUM every 3rd-4th substantive turn must be followed \
by a short backchannel turn from the listener. Both speakers backchannel. \
A backchannel is its own full turn, e.g.:
  CLIENT (Name): Mm-hmm.
  BROKER (Name): Right, right.
Acceptable backchannels: "Mm-hmm." / "Right." / "Yeah." / "Okay." / \
"Sure." / "Got it." / "Right, right." / "Yep." / "Uh-huh." / "Gotcha."

INTERRUPTIONS — include 1-3 per dialogue. The interrupted speaker's turn \
ends mid-sentence with an em dash:
  BROKER (Name): So the renewal is coming in at about fourteen—
  CLIENT (Name): Fourteen percent? You can't be serious.

DISFLUENCY — sprinkle throughout, every few turns:
- False starts: "So we — actually, hold on, let me back up."
- Self-corrections: "The deductible is five thou— sorry, twenty-five hundred."
- Filler clusters: "Um, so, yeah, basically..."
- Trailing off: "And that's where we'd want to..."
- Verbal tics: "you know", "I mean", "like", "right?"

INFORMATION DISCIPLINE — this is critical:
- Broker cites KNOWN items confidently
- On UNKNOWN items: broker hedges naturally ("let me check on that", \
"I'd need to pull that up", "I'll get back to you with specifics")
- On truly unexpected questions (not in KNOWN or UNKNOWN): broker says \
"that's a great question — I don't have that in front of me" or similar
- Broker NEVER fabricates specific numbers, forms, or dates not in KNOWN
- Include 1-2 client questions that surprise the broker — topics not \
listed in either KNOWN or UNKNOWN

CALL ARC:
- Outbound call: broker initiates with greeting, states purpose
- Follow the specified call outcome naturally
- Failure outcomes must feel organic — the client has real reasons
- Structure: greeting -> purpose -> main discussion -> resolution/next steps -> sign-off

Here is an example of proper backchanneling density and natural speech:

  BROKER (Name): So what I'm seeing on the renewal is the carrier came back \
at about fourteen percent over last year, which puts us at roughly—
  CLIENT (Name): Fourteen percent?
  BROKER (Name): Yeah, I know. It's, um — it's not great. So what I've done \
is I've gone ahead and gone out to two other markets to—
  CLIENT (Name): Mm-hmm.
  BROKER (Name): —to see what we can get. And one came back at around eight, \
which is still an increase but—
  CLIENT (Name): Right, right.
  BROKER (Name): —significantly better. The, uh, the trade-off is the network \
is a little narrower, so that's something we'd want to talk through.

CONTEXT INJECTIONS (when specified in the scenario):
Some calls include mid-call context injections — think of these as a senior \
broker or supervisor listening in and whispering coaching directions into the \
broker's ear during the call. Each injection is a SHORT, DIRECTIVE instruction \
that tells the broker what to DO, not just raw data.

When the scenario specifies N > 0 injections:
- Place exactly N [INJECT: ...] markers, each on its own line in the transcript
- Each injection is a directive: it gives the broker specific data AND tells \
them what to do with it. Format: action/recommendation + key facts. Examples:
  [INJECT: Quote the Berkshire alternative at $13,100 — 3.6% increase vs their current 12.3%. Emphasize the savings.]
  [INJECT: Claim #GL-2024-0847 reserve is $45K, adjuster Torres has site inspection 3/28. Reassure client timeline is normal.]
  [INJECT: Umbrella approved — $2M, $10K SIR, $3,400/yr. Push to bind today before underwriter hold expires Friday.]
  [INJECT: Client's Markel quote doesn't include completed ops tail. Flag that gap — it's a dealbreaker for municipal contracts.]
  [INJECT: Offer the 3-year rate lock at $11,800/yr. That beats their shopping quote and secures the account.]
- Injections must be SHORT — under 50 tokens. A senior's whispered coaching, \
not a data dump. Lead with the action, follow with just enough facts.
- PROACTIVE placement (~70% of injections): place the marker BEFORE a client \
turn where the client asks about the topic. The broker responds after the \
client speaks with grounded info and NO hedge — they already have it. This \
simulates the system anticipating the need and pre-loading during client speech.
- REACTIVE placement (~30% of injections): place the marker AFTER a broker \
hedge turn ("let me pull that up", "give me one sec"). The broker then \
responds with grounded info. This simulates a brief lookup pause.
- The broker MUST follow the injection's directive and cite ONLY the facts \
it provides — no made-up extras
- Injections should resolve UNKNOWN items (or surprise questions)

When the scenario specifies 0 injections:
- No [INJECT: ...] markers anywhere in the transcript
- Broker hedges on ALL unknown items and promises to follow up (never fabricates)

FORMAT:
- Wrap the pre-call brief in <brief>...</brief> tags
- Wrap the transcript in <transcript>...</transcript> tags
- Transcript starts with [Call start] and ends with [Call end]
- Speaker labels: BROKER (Name): and CLIENT (Name):
- [INJECT: ...] markers go on their own line between speaker turns
- Target: 200-1200 words depending on complexity (excluding injection markers)

Output the brief first, then the transcript. Nothing else outside the tags."""


def build_user_prompt(seed: dict) -> str:
    """Build the user prompt describing the outbound call scenario."""
    b = seed["broker"]
    c = seed["client"]
    out = seed["outcome"]

    lines = [
        "Generate an outbound insurance broker phone call with these parameters:",
        "",
        "=== BROKER ===",
        f"Name: {b['name']}",
        f"Experience: {b['experience_level']} — {b['experience_description']}",
        f"Specialty: {b['specialty']}",
        f"Communication style: {b['communication_style']}",
        f"Regional speech pattern: {b['regional_speech']}",
        "",
        "=== CLIENT ===",
        f"Name: {c['name']}",
        f"Company size: {c['company_size']} — {c['company_description']}",
        f"Role: {c['role']}",
        f"Insurance sophistication: {c['insurance_sophistication']} — {c['sophistication_description']}",
        f"Relationship with broker: {c['relationship_tenure']} — {c['tenure_description']}",
        "",
        "=== SCENARIO ===",
        f"Broker's outbound goal: {seed['goal']}",
        f"Call outcome: {out['label']} — {out['description']}",
        f"Emotional tenor: {seed['tenor']}",
        f"Primary domain vocabulary: {seed['domain_vocab']}",
        f"Complexity: {seed['complexity']} — {seed['complexity_description']}",
        f"Context injections: {seed['injection_count']}"
        + (" — include {} mid-call [INJECT: ...] marker{}".format(
            seed['injection_count'],
            "s" if seed['injection_count'] != 1 else "")
           if seed['injection_count'] > 0
           else " — no injections, broker works from system prompt only"),
        "",
        "Generate the pre-call brief and transcript now.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response Parsing
# ---------------------------------------------------------------------------

_SPEAKER_RE = re.compile(r'^(BROKER|CLIENT)\s*\([^)]*\):')
_INJECT_RE = re.compile(r'^\[INJECT:\s*(.*)\]$')


def parse_response(text: str) -> tuple[str | None, str | None, list[dict]]:
    """Extract pre-call brief, clean transcript, and injection metadata.

    Returns (brief, transcript_without_markers, context_injections).
    Each injection is {"after_turn": int, "text": str}.
    """
    brief_match = re.search(r'<brief>\s*(.*?)\s*</brief>', text, re.DOTALL)
    transcript_match = re.search(r'<transcript>\s*(.*?)\s*</transcript>', text, re.DOTALL)

    brief = brief_match.group(1).strip() if brief_match else None
    raw_transcript = transcript_match.group(1).strip() if transcript_match else None

    if raw_transcript is None:
        return brief, None, []

    # Walk lines: count speaker turns, extract [INJECT: ...] markers, build clean transcript
    injections = []
    clean_lines = []
    turn_count = 0

    for line in raw_transcript.split('\n'):
        stripped = line.strip()

        inject_match = _INJECT_RE.match(stripped)
        if inject_match:
            injections.append({
                "after_turn": max(turn_count - 1, 0),
                "text": inject_match.group(1).strip(),
            })
            continue  # strip marker from clean transcript

        clean_lines.append(line)
        if _SPEAKER_RE.match(stripped):
            turn_count += 1

    return brief, '\n'.join(clean_lines), injections


# ---------------------------------------------------------------------------
# Synchronous API Calls
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 16000


def generate_single(
    client: anthropic.Anthropic,
    seed: dict,
    dialogue_id: str,
    model: str,
    max_retries: int = 3,
    base_delay: float = 5.0,
) -> dict:
    """Call the Claude API for a single dialogue, with retries. Returns a record dict."""
    user_prompt = build_user_prompt(seed)
    now = datetime.now(timezone.utc).isoformat()

    for attempt in range(max_retries):
        try:
            log.info("[%s] calling API...", dialogue_id)
            response = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Extract text content (skip thinking blocks if present)
            text_parts = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
            full_text = "\n".join(text_parts)

            brief, transcript, injections = parse_response(full_text)

            if not brief or not transcript:
                log.warning("[%s] Parse failed — missing %s",
                            dialogue_id, "brief" if not brief else "transcript")
                return {
                    "id": dialogue_id,
                    "seed": seed,
                    "user_prompt": user_prompt,
                    "text_prompt": None,
                    "dialogue": None,
                    "context_injections": [],
                    "raw_response": full_text[:3000],
                    "model": model,
                    "error_type": "parse_error",
                    "error": f"Missing {'brief' if not brief else 'transcript'}",
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                    "generated_at": now,
                }

            expected = seed.get("injection_count", 0)
            actual = len(injections)
            if expected > 0 and actual != expected:
                log.warning("[%s] Expected %d injections, got %d",
                            dialogue_id, expected, actual)

            return {
                "id": dialogue_id,
                "seed": seed,
                "user_prompt": user_prompt,
                "text_prompt": brief,
                "dialogue": transcript,
                "context_injections": injections,
                "model": model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "generated_at": now,
            }

        except anthropic.RateLimitError:
            delay = base_delay * (2 ** attempt)
            log.warning("[%s] Rate limited (attempt %d/%d). Waiting %.0fs...",
                        dialogue_id, attempt + 1, max_retries, delay)
            time.sleep(delay)

        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                delay = base_delay * (2 ** attempt)
                log.warning("[%s] Server error %d (attempt %d/%d). Waiting %.0fs...",
                            dialogue_id, e.status_code, attempt + 1, max_retries, delay)
                time.sleep(delay)
            else:
                log.error("[%s] API error %d: %s", dialogue_id, e.status_code, e.message)
                return {
                    "id": dialogue_id,
                    "seed": seed,
                    "user_prompt": user_prompt,
                    "text_prompt": None,
                    "dialogue": None,
                    "model": model,
                    "error_type": f"api_error_{e.status_code}",
                    "error": str(e.message),
                    "generated_at": now,
                }

        except anthropic.APIConnectionError:
            delay = base_delay * (2 ** attempt)
            log.warning("[%s] Connection error (attempt %d/%d). Waiting %.0fs...",
                        dialogue_id, attempt + 1, max_retries, delay)
            time.sleep(delay)

    # All retries exhausted
    log.error("[%s] Failed after %d attempts.", dialogue_id, max_retries)
    return {
        "id": dialogue_id,
        "seed": seed,
        "user_prompt": user_prompt,
        "text_prompt": None,
        "dialogue": None,
        "model": model,
        "error_type": "max_retries_exhausted",
        "error": f"Failed after {max_retries} attempts",
        "generated_at": now,
    }


def generate_all(
    client: anthropic.Anthropic,
    seeds: list[dict],
    model: str,
    output_path: str,
    workers: int = 1,
) -> None:
    """Generate all dialogues, optionally in parallel, writing JSONL output."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    succeeded = 0
    errored = 0
    total = len(seeds)
    write_lock = threading.Lock()

    with open(output_path, "w", encoding="utf-8") as f:

        def _process(i: int, seed: dict) -> dict:
            dialogue_id = f"dial-{i:05d}"
            log.info("[%d/%d] Generating %s...", i + 1, total, dialogue_id)
            return generate_single(client, seed, dialogue_id, model)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process, i, seed): i
                for i, seed in enumerate(seeds)
            }

            for future in as_completed(futures):
                idx = futures[future]
                record = future.result()
                dialogue_id = record["id"]

                if record.get("dialogue") is not None:
                    succeeded += 1
                    tokens = record["usage"]
                    log.info("%s — OK (%d in / %d out tokens) [%d/%d done]",
                             dialogue_id,
                             tokens["input_tokens"], tokens["output_tokens"],
                             succeeded + errored, total)
                else:
                    errored += 1
                    log.error("%s — FAILED: %s [%d/%d done]",
                              dialogue_id, record.get("error", "unknown"),
                              succeeded + errored, total)

                with write_lock:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()

    log.info("Results written to %s — succeeded: %d, errored: %d (of %d total)",
             output_path, succeeded, errored, total)


# ---------------------------------------------------------------------------
# Dry Run
# ---------------------------------------------------------------------------

def dry_run(seeds: list[dict], model: str) -> None:
    """Print sample prompts without submitting to the API."""
    for i, seed in enumerate(seeds):
        dialogue_id = f"dial-{i:05d}"
        user_prompt = build_user_prompt(seed)
        print("=" * 80)
        print(f"ID: {dialogue_id}")
        print(f"Model: {model}")
        print(f"Max tokens: {MAX_TOKENS}")
        print(f"Outcome: {seed['outcome']['label']} — {seed['outcome']['description']}")
        print(f"Injections: {seed['injection_count']}")
        print("-" * 40)
        print("SYSTEM PROMPT (first 300 chars):")
        print(SYSTEM_PROMPT[:300] + "...")
        print("-" * 40)
        print("USER PROMPT:")
        print(user_prompt)
        print("-" * 40)
        print("SEED COMBINATION:")
        print(json.dumps(seed, indent=2, ensure_ascii=False))
        print("=" * 80)
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate diverse insurance broker outbound call transcripts via Claude API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python generate_dialogues_sync.py                          # 200 dialogues
  python generate_dialogues_sync.py -n 500 -o out/batch.jsonl -s 42
  python generate_dialogues_sync.py --dry-run -n 3           # preview prompts
""",
    )
    parser.add_argument(
        "-n", "--num-dialogues",
        type=int,
        default=200,
        help="Number of dialogues to generate (default: 200)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="dialogues.jsonl",
        help="Output JSONL file path (default: dialogues.jsonl)",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (default: random)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Claude model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=1,
        help="Number of parallel API calls (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sample prompts without calling the API",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build seed axes
    brokers = _build_broker_personas()
    clients = _build_client_personas()
    goals = BROKER_GOALS
    tenors = EMOTIONAL_TENORS
    vocabs = DOMAIN_VOCAB_CLUSTERS
    complexities = COMPLEXITY_LEVELS

    log.info(
        "Seed axes — brokers: %d, clients: %d, goals: %d, tenors: %d, "
        "vocabs: %d, complexities: %d, outcomes: %d (80/20 weighted)",
        len(brokers), len(clients), len(goals), len(tenors),
        len(vocabs), len(complexities),
        len(OUTCOMES_SUCCESS) + len(OUTCOMES_FAILURE),
    )
    cross_product = (
        len(brokers) * len(clients) * len(goals) * len(tenors)
        * len(vocabs) * len(complexities)
        * (len(OUTCOMES_SUCCESS) + len(OUTCOMES_FAILURE))
    )
    log.info("Cross-product space: %s combinations", f"{cross_product:,}")

    # LHS sampling — 8 dimensions (6 uniform + 1 weighted outcome + 1 weighted injection)
    rng = np.random.default_rng(args.seed)
    log.info("Generating %d LHS samples (seed=%s)...", args.num_dialogues, args.seed)
    lhs_matrix = latin_hypercube_sample(args.num_dialogues, 8, rng)
    seeds = map_samples_to_seeds(lhs_matrix, brokers, clients, goals, tenors, vocabs, complexities)
    log.info("Mapped %d seed combinations.", len(seeds))

    # Verify outcome distribution
    outcome_counts = {}
    for s in seeds:
        label = s["outcome"]["label"]
        outcome_counts[label] = outcome_counts.get(label, 0) + 1
    success_n = sum(v for k, v in outcome_counts.items() if "failure" not in k)
    failure_n = sum(v for k, v in outcome_counts.items() if "failure" in k)
    log.info("Outcome split — success: %d (%.0f%%), failure: %d (%.0f%%)",
             success_n, 100 * success_n / len(seeds),
             failure_n, 100 * failure_n / len(seeds))

    # Verify injection distribution
    inj_counts = {}
    for s in seeds:
        n = s["injection_count"]
        inj_counts[n] = inj_counts.get(n, 0) + 1
    with_inj = sum(v for k, v in inj_counts.items() if k > 0)
    without_inj = inj_counts.get(0, 0)
    log.info("Injection split — with: %d (%.0f%%), without: %d (%.0f%%) | counts: %s",
             with_inj, 100 * with_inj / len(seeds),
             without_inj, 100 * without_inj / len(seeds),
             dict(sorted(inj_counts.items())))

    if args.dry_run:
        dry_run(seeds, args.model)
        log.info("Dry run complete — no API calls made.")
        return

    # Generate dialogues
    api_client = anthropic.Anthropic()
    log.info("Starting generation with %d worker(s)...", args.workers)
    generate_all(api_client, seeds, args.model, args.output, workers=args.workers)

    log.info("Done. Output: %s", args.output)


if __name__ == "__main__":
    main()
