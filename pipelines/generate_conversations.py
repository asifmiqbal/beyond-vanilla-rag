"""Multi-Turn Synthetic Dialogue Generator (100K Conversations, 35-50 Turns).

Features:
- Strict 35 to 50 turns per conversation
- 3-actor coordination: customer, ai_agent, human_agent
- 5 realistic telco escalation arcs:
  1. Billing dispute & unauthorized roaming charge refund
  2. Complete fiber optic outage & technician truck roll dispatch
  3. International roaming failure & carrier steering reset
  4. SIM swap fraud alert & biometric KYC verification hold
  5. Enterprise / family 5G pooling plan upgrade negotiation
- Strict 5-state Finite State Machine (FSM):
  BOT_TRIAGE -> ESCALATION_TRIGGER -> WARM_HANDOFF -> HUMAN_ACTIVE -> RESOLVED
- Turn-level ground-truth intent, noul urgency, and frustration annotations
- Memory-efficient streaming into compressed JSONL batches (.jsonl.gz)
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

console = Console()

FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "Riley", "Casey", "Avery", "David", "Sarah", "Elena", "Marcus", "Li", "Kenji", "Fatima", "Carlos"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Wong", "Patel", "Tanaka", "Ivanov"]
CITIES = ["Dallas", "Seattle", "Chicago", "Atlanta", "Miami", "Denver", "Phoenix", "Boston", "San Jose", "Austin"]

SCENARIOS = [
    "billing_dispute_roaming",
    "fiber_outage_technician",
    "roaming_failure_overseas",
    "sim_swap_kyc_fraud",
    "plan_upgrade_enterprise",
]


def _generate_dialogue_turns(
    scenario: str,
    target_turns: int,
    conv_id: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generates strictly target_turns turns (35-50) using the 5-state FSM."""
    customer_name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    account_num = f"ACCT-{random.randint(100000, 999999)}"
    invoice_id = f"INV-{random.randint(10000, 99999)}"
    phone_num = f"+1-{random.randint(200, 999)}-555-{random.randint(1000, 9999)}"
    city = random.choice(CITIES)

    # Escalation trigger point is placed between turn 12 and 18
    escalation_turn = random.randint(12, 18)
    # Warm handoff occurs immediately at escalation_turn + 1
    warm_handoff_turn = escalation_turn + 1

    turns: List[Dict[str, Any]] = []

    # Scenario-specific dialogue building blocks
    if scenario == "billing_dispute_roaming":
        primary_intent = "billing_inquiry"
        triage_exchanges = [
            ("customer", f"Hi, I am looking at invoice #{invoice_id} for {account_num} and see an extra charge of $120. Can someone explain this?", 0.20, 0.40),
            ("ai_agent", f"Hello {customer_name}! I would be happy to review invoice #{invoice_id} for you. To access your ledger securely, could you please confirm your 4-digit security PIN?", 0.05, 0.10),
            ("customer", f"My PIN is {random.randint(1000, 9999)}. I had purchased a 7-day international pass before traveling to Europe.", 0.10, 0.35),
            ("ai_agent", f"Thank you for confirming your PIN. I see your pass TELCO-ROAM-Z1-DAY was activated on the 5th. However, data sessions in transit triggered standard roaming rates.", 0.05, 0.15),
            ("customer", "That makes no sense! The sales rep told me the pass covered maritime and airline roaming as well!", 0.25, 0.60),
            ("ai_agent", "I apologize for any misunderstanding. Zone 1 passes cover terrestrial EU networks, while maritime roaming requires Zone 4 add-on TELCO-ROAM-Z4-INFLIGHT.", 0.05, 0.20),
            ("customer", "Why did your app not warn me when connecting? I never agreed to pay $120 for 15 minutes of email syncing!", 0.30, 0.72),
            ("ai_agent", "Our network sends an automated SMS alert when latching onto international maritime carriers. Would you like me to review the connection timestamps?", 0.05, 0.25),
            ("customer", "I did not receive any SMS until 6 hours later! You cannot charge me for this lack of notification.", 0.35, 0.78),
            ("ai_agent", "Under standard billing policy, third-party carrier transit fees are passed through to the monthly invoice.", 0.05, 0.30),
            ("customer", f"This is completely unacceptable. You have charged my card on file without my consent. Reverse invoice #{invoice_id} now.", 0.40, 0.85),
            ("customer", "Stop giving me automated bot policy excuses! I demand to speak with a human manager or supervisor right now!", 0.82, 0.95),
        ]
        handoff_msg = f"I sincerely apologize for the frustration. I am transferring your account {account_num} to Marcus in our Senior Billing Escalation Desk along with your dispute notes."
        human_exchanges = [
            ("human_agent", f"Hello {customer_name}, my name is Marcus from Senior Billing Escalations. I have taken over your case regarding the $120 transit charge on #{invoice_id}.", 0.10, 0.30),
            ("customer", "Thank you, Marcus. Finally a human. The bot was just reciting policies while your system charged me $120 unfairly.", 0.15, 0.50),
            ("human_agent", "I completely understand your frustration. I am pulling up the carrier CDR logs for your European transit segment.", 0.05, 0.10),
            ("customer", "You will see that I had the Day Pass activated before boarding.", 0.10, 0.25),
            ("human_agent", "Indeed, I confirm TELCO-ROAM-Z1-DAY was active. There was an SMS delivery lag of 4 hours during the carrier handover in maritime waters.", 0.05, 0.10),
            ("customer", "Exactly! So how was I supposed to know the pass didn't apply?", 0.10, 0.20),
            ("human_agent", "You are completely correct. Since our notification gateway failed the SLA threshold, I am approving an immediate courtesy credit of $120.00.", 0.05, 0.05),
            ("customer", "Thank you so much! Will that credit show up on this month's statement?", 0.05, 0.05),
            ("human_agent", f"Yes, I have applied adjustment code ADJ-DISPUTE-ROAM directly to #{invoice_id}. Your revised balance due is $45.00.", 0.05, 0.05),
            ("customer", "That is such a relief. Do I need to do anything to prevent this on my return flight next week?", 0.05, 0.05),
            ("human_agent", "I have added a roaming network latch lock on your line so it will only connect to verified terrestrial partner networks.", 0.05, 0.05),
            ("customer", "That is perfect. Can you also send me an email receipt of the $120 credit?", 0.05, 0.05),
            ("human_agent", f"Receipt has just been dispatched to the email on file for {account_num}. Is there anything else I can assist you with today?", 0.05, 0.05),
            ("customer", "No, Marcus, you solved everything quickly. Thank you for your help!", 0.05, 0.05),
            ("human_agent", "It was my absolute pleasure. Thank you for choosing Apex Telecom, and have a safe trip home!", 0.05, 0.05),
        ]

    elif scenario == "fiber_outage_technician":
        primary_intent = "fiber_broadband"
        triage_exchanges = [
            ("customer", f"URGENT: Our fiber internet is completely down at our office in {city}. The optical box has a red light!", 0.85, 0.65),
            ("ai_agent", f"Hello {customer_name}! I can help troubleshoot your FTTH connection. Is the red light on the 'LOS' (Loss of Signal) indicator?", 0.20, 0.15),
            ("customer", "Yes! LOS is blinking red, and the PON light is completely off. We have 15 people unable to work!", 0.90, 0.75),
            ("ai_agent", "A blinking red LOS LED indicates an optical signal drop below -28 dBm. Let's inspect the yellow fiber patch cord. Has it been bent or disconnected?", 0.30, 0.20),
            ("customer", "No, nobody touched the ONT. The cable is firmly clicked in. We restarted the modem twice already.", 0.85, 0.80),
            ("ai_agent", "I am running an automated optical line loopback test on your GPON circuit from our central distribution frame.", 0.20, 0.15),
            ("customer", "Please hurry, our business operations are completely paralyzed right now!", 0.95, 0.88),
            ("ai_agent", "The loopback test timed out with error ERR_ONT_LOS_RED. Telemetry reports an open circuit 420 meters from your building.", 0.40, 0.25),
            ("customer", "There is construction work down the street! A backhoe must have cut the fiber drop line!", 0.92, 0.85),
            ("ai_agent", "If physical civil excavation caused a fiber cut, local hardware rebooting will not restore service.", 0.20, 0.30),
            ("customer", "Then dispatch a technician immediately! We pay for business SLA! Transfer me to your dispatch manager NOW!", 0.98, 0.95),
            ("customer", "DO NOT tell me to wait 48 hours! Transfer me to an emergency dispatch supervisor!", 0.99, 0.98),
        ]
        handoff_msg = f"Connecting you immediately to Tier-2 Emergency Broadband Dispatch with high priority ticket INC-FTTH-{random.randint(1000, 9999)}."
        human_exchanges = [
            ("human_agent", f"Emergency Dispatch desk, this is Officer Dave. I see the optical break at -420m on your GPON circuit {account_num}.", 0.70, 0.40),
            ("customer", "Dave, thank goodness. Construction crews just dug up the street corner. We need a splicing team right away.", 0.75, 0.60),
            ("human_agent", "I am looking at our regional field map in {city}. We already have a fiber repair truck en route to 4th & Elm Street.", 0.30, 0.20),
            ("customer", "That is right on our block! Can they verify our drop cable?", 0.40, 0.20),
            ("human_agent", "Yes, I have attached your circuit to priority ticket INC-FIBER-EMERGENCY. Technician Alex has been assigned specifically to your premises drop.", 0.10, 0.10),
            ("customer", "What is their estimated arrival time?", 0.30, 0.15),
            ("human_agent", "Alex's GPS puts him 18 minutes away. He will have OTDR optical splicing tools on site.", 0.10, 0.10),
            ("customer", "Will he need access inside our server closet?", 0.20, 0.10),
            ("human_agent", "Only if the outdoor splice test requires terminal calibration. Please keep someone available at the main door.", 0.05, 0.05),
            ("customer", "Understood. Our IT manager Sam will meet him at the front desk.", 0.10, 0.05),
            ("human_agent", f"I have logged Sam's contact info. In the interim, I am provisioning temporary 5G failover data passes for your staff.", 0.05, 0.05),
            ("customer", "Wow, that is incredibly helpful. How do we activate the 5G failover?", 0.05, 0.05),
            ("human_agent", "I have authorized 100GB emergency tethering pool on all 5 registered business lines for {account_num}.", 0.05, 0.05),
            ("customer", "That will keep our core team online until the fiber is spliced. Thank you, Dave.", 0.05, 0.05),
            ("human_agent", "You are very welcome. I will monitor your optical dBm levels remotely and send an SMS the moment light levels return to -19 dBm.", 0.05, 0.05),
        ]

    elif scenario == "roaming_failure_overseas":
        primary_intent = "roaming_passes"
        triage_exchanges = [
            ("customer", f"I just landed in Tokyo and my phone has zero data service! Account #{account_num}. Please help, I cannot call my cab!", 0.88, 0.70),
            ("ai_agent", f"Hello {customer_name}! I can troubleshoot your international roaming right away. Do you see any signal bars or carrier name on your screen?", 0.20, 0.10),
            ("customer", "It says 'Emergency Calls Only' and won't connect to Docomo or Softbank at all.", 0.85, 0.75),
            ("ai_agent", "Please check: Is 'Data Roaming' toggled ON under your device Cellular Data settings?", 0.10, 0.10),
            ("customer", "Yes, Data Roaming has been enabled since before take-off. I have zero internet.", 0.80, 0.78),
            ("ai_agent", "Let's check your APN. Please confirm your APN is set to 'roam.telco.apac' with APN protocol IPv4/IPv6.", 0.10, 0.10),
            ("customer", "I checked, APN is roam.telco.apac. Still getting error ERR_ROAM_AUTH_403!", 0.85, 0.82),
            ("ai_agent", "Error ERR_ROAM_AUTH_403 indicates our home subscriber server (HSS) is rejecting foreign carrier steering handshake.", 0.20, 0.15),
            ("customer", "I am stranded at Narita airport with 5% battery! I need this fixed immediately!", 0.95, 0.90),
            ("ai_agent", "I can submit a standard HSS IMSI refresh which takes approximately 30 minutes to propagate.", 0.20, 0.35),
            ("customer", "30 minutes?! My phone will die in 5 minutes! Connect me to a real engineer right now!", 0.98, 0.95),
            ("customer", "GET ME A HUMAN AGENT BEFORE MY BATTERY DIES!", 0.99, 0.99),
        ]
        handoff_msg = "Routing directly to our Global Roaming Operations desk with Emergency Roaming Priority."
        human_exchanges = [
            ("human_agent", f"Global Roaming NOC, this is Kenji. {customer_name}, I am forcing an instant OTA network registration reset for your IMSI.", 0.30, 0.20),
            ("customer", "Kenji, please hurry, my battery is at 4%.", 0.80, 0.60),
            ("human_agent", "Reset signal sent. Toggle Airplane Mode ON for 5 seconds, then toggle OFF right now.", 0.10, 0.10),
            ("customer", "Done... Waiting... It says searching...", 0.40, 0.25),
            ("human_agent", "I see your SIM handshaking with NTT DOCOMO LTE node #8904. Signal is strong.", 0.05, 0.05),
            ("customer", "YES! The 5G icon just popped up and my WhatsApp messages are downloading!", 0.10, 0.10),
            ("human_agent", "Fantastic. I have also manually pinned your profile to Softbank and Docomo so your phone won't hunt for weak partner towers.", 0.05, 0.05),
            ("customer", "You literally saved my trip. I can book my taxi now.", 0.05, 0.05),
            ("human_agent", f"I also verified your Zone 3 Roaming Pass is active with 15GB high-speed data remaining. No extra fees will apply.", 0.05, 0.05),
            ("customer", "Thank you so much Kenji. Excellent service.", 0.05, 0.05),
            ("human_agent", "My pleasure! Safe travels in Japan!", 0.05, 0.05),
        ]

    elif scenario == "sim_swap_kyc_fraud":
        primary_intent = "account_security"
        triage_exchanges = [
            ("customer", f"SECURITY ALERT: I received an SMS saying a SIM swap was requested for my number {phone_num}! I DID NOT REQUEST THIS!", 0.95, 0.85),
            ("ai_agent", f"Hello {customer_name}. We take unauthorized SIM swap requests very seriously. Let me immediately freeze your line {phone_num}.", 0.40, 0.20),
            ("customer", "Please freeze it now! I think someone has my password or compromised my email!", 0.95, 0.90),
            ("ai_agent", "Line lock has been initiated under fraud protocol SEC-FRAUD-HOLD. An OTP has been sent to your backup email.", 0.30, 0.20),
            ("customer", "The hacker might have access to my email! Do not trust email OTP!", 0.98, 0.95),
            ("ai_agent", "Under our security standards, OTP verification is required to cancel an eSIM provisioning order.", 0.30, 0.40),
            ("customer", "Are you crazy?! The hacker is redirecting my email! Put me through to your fraud investigation team immediately!", 0.99, 0.99),
            ("customer", "I am a victim of identity theft right this second! TRANSFER ME TO A FRAUD AGENT!", 0.99, 1.00),
        ]
        handoff_msg = "Critical security alert confirmed. Transferring immediately to Tier-2 Fraud & Cybersecurity Specialists."
        human_exchanges = [
            ("human_agent", f"Fraud Prevention Team, Senior Investigator Sarah speaking. {customer_name}, your line {phone_num} is now in hard cryptographic lockdown.", 0.20, 0.20),
            ("customer", "Thank god. Sarah, did the new eSIM activate?", 0.50, 0.40),
            ("human_agent", "No, the pending QR provisioning was caught in our 4-hour cooling-off window. I have permanently revoked the rogue QR profile.", 0.05, 0.05),
            ("customer", "Thank goodness. How did they initiate it in the first place?", 0.20, 0.15),
            ("human_agent", "They attempted an online portal password reset using leaked credentials. I have terminated all active web sessions across all devices.", 0.05, 0.05),
            ("customer", "What do I need to do to secure my account now?", 0.10, 0.10),
            ("human_agent", "We will perform a live government ID verification via our encrypted portal right now to reset your Master PIN.", 0.05, 0.05),
            ("customer", "I have my driver's license ready. Sending the photo now.", 0.05, 0.05),
            ("human_agent", "Received and verified. Your new 6-digit verbally verified security PIN is now active.", 0.05, 0.05),
            ("customer", "Sarah, you prevented a total disaster for my bank accounts. Thank you.", 0.05, 0.05),
            ("human_agent", "That is what we are here for. We also added mandatory biometric in-store authorization for any future SIM changes on your profile.", 0.05, 0.05),
        ]

    else:  # plan_upgrade_enterprise
        primary_intent = "plan_upgrade"
        triage_exchanges = [
            ("customer", f"Hello, we have 25 lines on our business account #{account_num} and want to upgrade everyone to 5G Unlimited with shared pooling.", 0.10, 0.10),
            ("ai_agent", f"Welcome {customer_name}! I can certainly assist with corporate plan upgrades. Our TELCO-ENTERPRISE-POOL tier offers 500GB shared data for 25 lines.", 0.05, 0.05),
            ("customer", "What is the monthly pricing per line on that tier?", 0.05, 0.05),
            ("ai_agent", "The standard rate is $45 per line per month, including unlimited domestic talk, text, and 5G Ultra-Wideband access.", 0.05, 0.05),
            ("customer", "Competitor carrier offered us $32 per line with free international roaming in Canada and Mexico.", 0.15, 0.30),
            ("ai_agent", "Our standard system discount caps at $40 per line for volume under 50 lines.", 0.05, 0.10),
            ("customer", "We have been with Apex Telecom for 8 years. $40 is not competitive enough. We are considering porting all 25 numbers out.", 0.40, 0.65),
            ("ai_agent", "I understand retention value is important. I can apply a one-time bill credit of $200 across the account.", 0.05, 0.15),
            ("customer", "A one-time $200 credit on a $12,000 annual contract is insulting. Connect me to your Corporate Retention Director.", 0.60, 0.85),
            ("customer", "Either I speak with a commercial account executive who has real discount authority, or we submit port-out requests today.", 0.75, 0.90),
        ]
        handoff_msg = "Transferring your commercial account to our Senior Enterprise Retention Specialist."
        human_exchanges = [
            ("human_agent", f"Good afternoon {customer_name}, this is Elena from Enterprise Accounts. I see you manage 25 lines on #{account_num} and are reviewing renewals.", 0.10, 0.20),
            ("customer", "Hello Elena. As I told the bot, we have a firm offer from your competitor at $32/line with North American roaming included.", 0.15, 0.35),
            ("human_agent", "We deeply value your 8-year partnership with Apex. I have direct authorization to match that rate.", 0.05, 0.05),
            ("customer", "Can you do $32 per line with Mexico and Canada roaming included?", 0.05, 0.05),
            ("human_agent", "Yes. I can bundle plan TELCO-ENTERPRISE-5G-NA at $31.50 per line on a 24-month corporate agreement.", 0.05, 0.05),
            ("customer", "What about device upgrades? Several of our sales reps need new 5G handsets.", 0.05, 0.05),
            ("human_agent", "I can provide 10 complimentary 5G enterprise devices with trade-in waiver as part of this agreement.", 0.05, 0.05),
            ("customer", "That is an excellent deal Elena. How soon can you send over the digital signature contract?", 0.05, 0.05),
            ("human_agent", "The contract package will arrive in your inbox in 10 minutes. Once countersigned, the new rates take effect on tomorrow's billing cycle.", 0.05, 0.05),
            ("customer", "Appreciate the quick work and fair negotiation.", 0.05, 0.05),
            ("human_agent", "Thank you for your loyalty to Apex Telecom. Have a productive week!", 0.05, 0.05),
        ]

    # Assemble and pad to target_turns (35 to 50)
    # 1. Triage turns
    for role, text, urg, frus in triage_exchanges[:escalation_turn]:
        turns.append({
            "turn_id": len(turns) + 1,
            "role": role,
            "content": text,
            "ground_truth_intent": "human_escalation" if frus >= 0.80 and role == "customer" else primary_intent,
            "is_urgent": urg,
            "frustration_score": frus,
            "escalation_status": "handover_triggered" if frus >= 0.80 or urg >= 0.75 else "bot_contained",
        })

    # 2. Warm Handoff turn (AI or System)
    turns.append({
        "turn_id": len(turns) + 1,
        "role": "ai_agent",
        "content": handoff_msg,
        "ground_truth_intent": primary_intent,
        "is_urgent": 0.10,
        "frustration_score": 0.20,
        "escalation_status": "warm_handoff",
    })

    # 3. Human Agent active turns
    for role, text, urg, frus in human_exchanges:
        if len(turns) >= target_turns - 4:
            break
        turns.append({
            "turn_id": len(turns) + 1,
            "role": role,
            "content": text,
            "ground_truth_intent": primary_intent,
            "is_urgent": urg,
            "frustration_score": frus,
            "escalation_status": "human_active",
        })

    # 4. Fill remaining turns naturally with verification/satisfaction until target_turns is reached
    resolution_phrases = [
        ("human_agent", "I have updated our internal tracking notes with case ID REF-OK-9912.", 0.05, 0.02),
        ("customer", "Could you also confirm the next billing date for our records?", 0.05, 0.02),
        ("human_agent", "Your standard billing cycle closes on the 28th of each calendar month.", 0.05, 0.02),
        ("customer", "Got it. And our customer care pin remains unchanged?", 0.05, 0.02),
        ("human_agent", "Yes, your PIN and primary security questions remain completely secure.", 0.05, 0.02),
        ("customer", "Understood. The explanation was very clear.", 0.05, 0.02),
        ("human_agent", "We strive to deliver five-star service on every escalation.", 0.05, 0.02),
        ("customer", "I will leave a positive rating on the post-call survey.", 0.05, 0.02),
        ("human_agent", "We truly appreciate that feedback. Have a wonderful rest of your day!", 0.05, 0.01),
        ("customer", "Thank you, goodbye!", 0.05, 0.01),
    ]

    phrase_idx = 0
    while len(turns) < target_turns:
        role, text, urg, frus = resolution_phrases[phrase_idx % len(resolution_phrases)]
        turns.append({
            "turn_id": len(turns) + 1,
            "role": role,
            "content": text,
            "ground_truth_intent": primary_intent,
            "is_urgent": urg,
            "frustration_score": frus,
            "escalation_status": "resolved" if len(turns) >= target_turns - 2 else "human_active",
        })
        phrase_idx += 1

    metadata = {
        "conversation_id": conv_id,
        "scenario": scenario,
        "customer_name": customer_name,
        "account_number": account_num,
        "total_turns": len(turns),
        "actors_involved": ["customer", "ai_agent", "human_agent"],
        "escalated": True,
        "escalation_turn": warm_handoff_turn,
        "resolution_status": "resolved_by_human_agent",
    }

    return turns, metadata


def generate_conversations_dataset(
    total_conversations: int = 100000,
    batch_size: int = 10000,
    output_dir: str = "data/synthetic",
) -> List[Path]:
    """Generates 100,000 synthetic dialogues streamed into compressed JSONL batches (.jsonl.gz)."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]Starting Synthetic Dialogue Generator[/bold cyan]")
    console.print(f"Target Volume: [bold yellow]{total_conversations:,}[/bold yellow] conversations | Batch Size: [yellow]{batch_size:,}[/yellow] | Turn Depth: [green]35-50[/green]")
    console.print(f"Output Directory: [dim]{out_path}[/dim]")

    generated_files: List[Path] = []
    num_batches = (total_conversations + batch_size - 1) // batch_size
    conv_counter = 0

    start_time = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total} dialogues)"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[green]Synthesizing conversations...", total=total_conversations)

        for batch_idx in range(num_batches):
            batch_target = min(batch_size, total_conversations - conv_counter)
            batch_filename = out_path / f"conversations_batch_{batch_idx+1:03d}.jsonl.gz"

            with gzip.open(batch_filename, "wt", encoding="utf-8") as gz_file:
                for b_i in range(batch_target):
                    conv_counter += 1
                    conv_id = f"TELCO-CONV-2026-{conv_counter:07d}"
                    scenario = SCENARIOS[conv_counter % len(SCENARIOS)]
                    
                    # Strictly between 35 and 50 turns
                    target_turns = random.randint(35, 50)

                    turns, meta = _generate_dialogue_turns(
                        scenario=scenario,
                        target_turns=target_turns,
                        conv_id=conv_id,
                    )

                    record = {**meta, "turns": turns}
                    gz_file.write(json.dumps(record) + "\n")
                    progress.update(task, advance=1)

            file_size_mb = os.path.getsize(batch_filename) / (1024 * 1024)
            generated_files.append(batch_filename)
            console.print(f"  [dim]Saved Batch {batch_idx+1:03d}/{num_batches:03d}: {batch_filename.name} ({file_size_mb:.2f} MB)[/dim]")

    total_duration = time.perf_counter() - start_time
    rate = total_conversations / max(total_duration, 0.001)
    console.print(f"[bold green]Generation Completed:[/bold green] {total_conversations:,} dialogues generated in [cyan]{total_duration:.2f}s[/cyan] ([bold yellow]{rate:,.0f} dialogues/sec[/bold yellow])")
    console.print(f"Total Batches Written: [yellow]{len(generated_files)}[/yellow] compressed JSONL archives.")

    return generated_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Turn Synthetic Dialogue Generator (35-50 Turns, 3 Actors)")
    parser.add_argument("--total", type=int, default=100000, help="Total conversations to synthesize (default: 100000)")
    parser.add_argument("--batch-size", type=int, default=10000, help="Conversations per compressed batch (default: 10000)")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory")

    args = parser.parse_args()
    generate_conversations_dataset(
        total_conversations=args.total,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
