"""
optimizer.py — Satisfaction-First Optimization Engine
=====================================================
Hyperparameter tuning and model optimization framework.
Uses PuLP (Linear Programming) to generate an optimal 24-hour activity schedule
that minimizes electricity costs while respecting time-window constraints
and user priorities.

Usage:
    from optimizer import generate_optimal_plan, Activity

    activities = [
        Activity("Dishwasher", power_kw=1.8, duration_hours=2, priority=3,
                 earliest_start=8, latest_finish=22),
        Activity("EV Charging", power_kw=7.4, duration_hours=4, priority=2,
                 earliest_start=22, latest_finish=8),  # overnight
    ]

    plan = generate_optimal_plan(activities, price_forecast, consumption_forecast)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from pulp import (
        LpProblem, LpMinimize, LpVariable, LpBinary,
        lpSum, LpStatus, value
    )
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    logger.warning("PuLP not installed. Install with: pip install PuLP")


# ═════════════════════════════════════════════════════════════
# Data Classes
# ═════════════════════════════════════════════════════════════

@dataclass
class Activity:
    """A household activity to be scheduled."""
    name: str
    power_kw: float           # Power consumption in kW
    duration_hours: int       # Duration in whole hours
    priority: int             # 1 (lowest) to 5 (highest)
    earliest_start: int       # Earliest start hour (0-23)
    latest_finish: int        # Latest finish hour (0-23, wraps around midnight)
    icon: str = "⚡"          # Display emoji


@dataclass
class ScheduledActivity:
    """A scheduled activity in the optimal plan."""
    name: str
    start_hour: int
    end_hour: int
    power_kw: float
    duration_hours: int
    priority: int
    cost_optimized: float     # Cost at optimal time (€)
    cost_baseline: float      # Cost if run at earliest possible time (€)
    savings: float            # Savings from optimization (€)
    icon: str = "⚡"


@dataclass
class OptimalPlan:
    """The complete optimized 24-hour schedule."""
    schedule: List[ScheduledActivity]
    hourly_power_profile: List[float]          # kW at each hour (0-23)
    hourly_price: List[float]                  # Price at each hour (€/MWh)
    hourly_consumption_forecast: List[float]   # Predicted consumption (MW)
    total_cost_optimized: float                # Total cost with optimization (€)
    total_cost_baseline: float                 # Total cost without optimization (€)
    total_savings: float                       # Total saved (€)
    savings_percentage: float                  # Savings as %
    forecasted_minimum_bill: float             # Absolute minimum bill possible (€)
    cheapest_hour: int                         # Hour with lowest price
    most_expensive_hour: int                   # Hour with highest price
    solver_status: str                         # LP solver status


# ═════════════════════════════════════════════════════════════
# Default Activity Presets
# ═════════════════════════════════════════════════════════════

DEFAULT_ACTIVITIES = [
    {
        "name": "Dishwasher",
        "power_kw": 1.8,
        "duration_hours": 2,
        "priority": 3,
        "earliest_start": 6,
        "latest_finish": 23,
        "icon": "🍽️"
    },
    {
        "name": "Washing Machine",
        "power_kw": 2.0,
        "duration_hours": 2,
        "priority": 3,
        "earliest_start": 6,
        "latest_finish": 22,
        "icon": "👕"
    },
    {
        "name": "Tumble Dryer",
        "power_kw": 2.5,
        "duration_hours": 2,
        "priority": 2,
        "earliest_start": 6,
        "latest_finish": 23,
        "icon": "🌀"
    },
    {
        "name": "EV Charging",
        "power_kw": 7.4,
        "duration_hours": 4,
        "priority": 2,
        "earliest_start": 20,
        "latest_finish": 8,
        "icon": "🔌"
    },
    {
        "name": "Cooking (Oven)",
        "power_kw": 2.5,
        "duration_hours": 2,
        "priority": 5,
        "earliest_start": 11,
        "latest_finish": 15,
        "icon": "🍳"
    },
    {
        "name": "Pool Pump",
        "power_kw": 1.5,
        "duration_hours": 3,
        "priority": 1,
        "earliest_start": 0,
        "latest_finish": 23,
        "icon": "🏊"
    },
    {
        "name": "Water Heater",
        "power_kw": 3.0,
        "duration_hours": 2,
        "priority": 4,
        "earliest_start": 5,
        "latest_finish": 9,
        "icon": "🚿"
    },
    {
        "name": "Air Conditioning",
        "power_kw": 2.0,
        "duration_hours": 6,
        "priority": 4,
        "earliest_start": 12,
        "latest_finish": 22,
        "icon": "❄️"
    }
]


def _get_valid_start_hours(activity: Activity) -> List[int]:
    """
    Get valid start hours for an activity, handling midnight wrap-around.
    E.g., earliest_start=22, latest_finish=8 means the window wraps midnight.
    """
    es = activity.earliest_start
    lf = activity.latest_finish
    dur = activity.duration_hours

    valid = []

    if es <= lf:
        # Normal window (e.g., 6-22)
        for h in range(es, lf - dur + 2):
            if h >= 0 and h + dur <= 24:
                valid.append(h)
    else:
        # Wrap-around window (e.g., 22-8 means 22,23,0,1,...,8)
        for h in range(es, 24):
            end = h + dur
            if end <= 24 or (end % 24) <= lf + 1:
                valid.append(h)
        for h in range(0, lf - dur + 2):
            if h >= 0 and h + dur <= 24:
                valid.append(h)

    if not valid:
        # Fallback: allow any hour where the task fits
        for h in range(0, 24 - dur + 1):
            valid.append(h)

    return sorted(set(valid))


def _compute_activity_cost(start_hour: int, duration: int,
                           power_kw: float, prices: List[float]) -> float:
    """
    Compute the cost of running an activity at a given start hour.
    Cost = sum of (price_per_kwh * power_kw) for each hour of operation.
    Prices are in €/MWh, so divide by 1000 to get €/kWh.
    """
    cost = 0.0
    for h_offset in range(duration):
        h = (start_hour + h_offset) % 24
        cost += (prices[h] / 1000.0) * power_kw  # €/kWh * kW * 1h = €
    return cost


# ═════════════════════════════════════════════════════════════
# Main Optimization Function
# ═════════════════════════════════════════════════════════════

def generate_optimal_plan(
    activities: List[Activity],
    price_forecast: List[float],
    consumption_forecast: List[float],
    max_total_power_kw: Optional[float] = None
) -> OptimalPlan:
    """
    Generate an optimal 24-hour activity schedule using Linear Programming.

    Inputs:
        - activities: List of Activity objects to schedule
        - price_forecast: 24-element list/np.array of predicted prices (€/MWh, index=hour)
        - consumption_forecast: 24-element list/np.array of predicted consumption (MW)
        - max_total_power_kw: Optional max concurrent power limit (kW)

    Outputs:
        - OptimalPlan with the optimal mapping schedule and grid constraint evaluation
        
    ML Principle: Constrained optimization interpreting predictive forecast outcomes to minimize
    objective function scores representing household cost.
    """
    assert len(price_forecast) == 24, "Price forecast must have 24 hourly values"
    assert len(consumption_forecast) == 24, "Consumption forecast must have 24 values"

    prices = list(price_forecast)
    cheapest_hour = int(np.argmin(prices))
    most_expensive_hour = int(np.argmax(prices))

    if not activities:
        return OptimalPlan(
            schedule=[],
            hourly_power_profile=[0.0] * 24,
            hourly_price=prices,
            hourly_consumption_forecast=list(consumption_forecast),
            total_cost_optimized=0.0,
            total_cost_baseline=0.0,
            total_savings=0.0,
            savings_percentage=0.0,
            forecasted_minimum_bill=0.0,
            cheapest_hour=cheapest_hour,
            most_expensive_hour=most_expensive_hour,
            solver_status="No activities"
        )

    if not PULP_AVAILABLE:
        return _greedy_fallback(activities, prices, consumption_forecast)

    # ─────────────────────────────────────────
    # LP Model Setup
    # ─────────────────────────────────────────
    prob = LpProblem("EnergyOptimizer", LpMinimize)

    # Decision variables: x[a][h] = 1 if activity a starts at hour h
    x = {}
    for i, act in enumerate(activities):
        valid_hours = _get_valid_start_hours(act)
        for h in valid_hours:
            x[i, h] = LpVariable(f"x_{i}_{h}", cat=LpBinary)

    # ─── Objective: Minimize total electricity cost ───
    prob += lpSum(
        x[i, h] * _compute_activity_cost(h, act.duration_hours, act.power_kw, prices)
        for i, act in enumerate(activities)
        for h in _get_valid_start_hours(act)
    ), "TotalCost"

    # ─── Constraint 1: Each activity scheduled exactly once ───
    for i, act in enumerate(activities):
        valid_hours = _get_valid_start_hours(act)
        prob += lpSum(x[i, h] for h in valid_hours) == 1, f"Schedule_{i}_{act.name}"

    # ─── Constraint 2: Max concurrent power (optional) ───
    if max_total_power_kw is not None:
        for t in range(24):
            power_at_t = []
            for i, act in enumerate(activities):
                valid_hours = _get_valid_start_hours(act)
                for h in valid_hours:
                    # Check if activity starting at h is running at time t
                    for offset in range(act.duration_hours):
                        if (h + offset) % 24 == t:
                            power_at_t.append(x[i, h] * act.power_kw)
            if power_at_t:
                prob += lpSum(power_at_t) <= max_total_power_kw, f"MaxPower_{t}"

    # ─── Solve ───
    logger.info("Solving LP optimization problem...")
    prob.solve()
    status = LpStatus[prob.status]
    logger.info(f"Solver status: {status}")

    if status != "Optimal":
        logger.warning(f"LP solver returned non-optimal status: {status}")
        return _greedy_fallback(activities, prices, consumption_forecast)

    # ─── Extract solution ───
    schedule = []
    hourly_power = [0.0] * 24

    for i, act in enumerate(activities):
        valid_hours = _get_valid_start_hours(act)
        for h in valid_hours:
            if value(x[i, h]) and value(x[i, h]) > 0.5:
                # This activity starts at hour h
                cost_opt = _compute_activity_cost(h, act.duration_hours, act.power_kw, prices)
                cost_base = _compute_activity_cost(
                    act.earliest_start, act.duration_hours, act.power_kw, prices
                )
                end_hour = (h + act.duration_hours) % 24

                scheduled = ScheduledActivity(
                    name=act.name,
                    start_hour=h,
                    end_hour=end_hour,
                    power_kw=act.power_kw,
                    duration_hours=act.duration_hours,
                    priority=act.priority,
                    cost_optimized=round(cost_opt, 4),
                    cost_baseline=round(cost_base, 4),
                    savings=round(cost_base - cost_opt, 4),
                    icon=act.icon
                )
                schedule.append(scheduled)

                # Update hourly power profile
                for offset in range(act.duration_hours):
                    hour_idx = (h + offset) % 24
                    hourly_power[hour_idx] += act.power_kw

                break  # Found the assigned hour

    # Sort schedule by start hour
    schedule.sort(key=lambda s: s.start_hour)

    total_opt = sum(s.cost_optimized for s in schedule)
    total_base = sum(s.cost_baseline for s in schedule)
    total_savings = total_base - total_opt
    savings_pct = (total_savings / total_base * 100) if total_base > 0 else 0

    # Calculate forecasted minimum bill (if all activities ran at cheapest possible hours)
    min_bill = 0.0
    for act in activities:
        best_cost = float("inf")
        for h in range(24 - act.duration_hours + 1):
            c = _compute_activity_cost(h, act.duration_hours, act.power_kw, prices)
            best_cost = min(best_cost, c)
        min_bill += best_cost

    return OptimalPlan(
        schedule=schedule,
        hourly_power_profile=hourly_power,
        hourly_price=prices,
        hourly_consumption_forecast=list(consumption_forecast),
        total_cost_optimized=round(total_opt, 4),
        total_cost_baseline=round(total_base, 4),
        total_savings=round(total_savings, 4),
        savings_percentage=round(savings_pct, 2),
        forecasted_minimum_bill=round(min_bill, 4),
        cheapest_hour=cheapest_hour,
        most_expensive_hour=most_expensive_hour,
        solver_status=status
    )


# ═════════════════════════════════════════════════════════════
# Greedy Fallback (if PuLP is not available)
# ═════════════════════════════════════════════════════════════

def _greedy_fallback(
    activities: List[Activity],
    prices: List[float],
    consumption_forecast: List[float]
) -> OptimalPlan:
    """
    Greedy scheduling fallback: schedule each activity at its cheapest valid hour.
    Used when PuLP solver is not available or fails.
    """
    logger.info("Using greedy fallback scheduler")

    cheapest_hour = int(np.argmin(prices))
    most_expensive_hour = int(np.argmax(prices))
    schedule = []
    hourly_power = [0.0] * 24

    # Sort by priority (high first) to give them first pick
    sorted_acts = sorted(activities, key=lambda a: a.priority, reverse=True)

    for act in sorted_acts:
        valid_hours = _get_valid_start_hours(act)
        best_h = valid_hours[0]
        best_cost = float("inf")

        for h in valid_hours:
            cost = _compute_activity_cost(h, act.duration_hours, act.power_kw, prices)
            if cost < best_cost:
                best_cost = cost
                best_h = h

        cost_base = _compute_activity_cost(
            act.earliest_start, act.duration_hours, act.power_kw, prices
        )

        end_hour = (best_h + act.duration_hours) % 24
        scheduled = ScheduledActivity(
            name=act.name,
            start_hour=best_h,
            end_hour=end_hour,
            power_kw=act.power_kw,
            duration_hours=act.duration_hours,
            priority=act.priority,
            cost_optimized=round(best_cost, 4),
            cost_baseline=round(cost_base, 4),
            savings=round(cost_base - best_cost, 4),
            icon=act.icon
        )
        schedule.append(scheduled)

        for offset in range(act.duration_hours):
            hour_idx = (best_h + offset) % 24
            hourly_power[hour_idx] += act.power_kw

    schedule.sort(key=lambda s: s.start_hour)
    total_opt = sum(s.cost_optimized for s in schedule)
    total_base = sum(s.cost_baseline for s in schedule)
    total_savings = total_base - total_opt
    savings_pct = (total_savings / total_base * 100) if total_base > 0 else 0

    min_bill = 0.0
    for act in activities:
        best_cost = float("inf")
        for h in range(24 - act.duration_hours + 1):
            c = _compute_activity_cost(h, act.duration_hours, act.power_kw, prices)
            best_cost = min(best_cost, c)
        min_bill += best_cost

    return OptimalPlan(
        schedule=schedule,
        hourly_power_profile=hourly_power,
        hourly_price=prices,
        hourly_consumption_forecast=list(consumption_forecast),
        total_cost_optimized=round(total_opt, 4),
        total_cost_baseline=round(total_base, 4),
        total_savings=round(total_savings, 4),
        savings_percentage=round(savings_pct, 2),
        forecasted_minimum_bill=round(min_bill, 4),
        cheapest_hour=cheapest_hour,
        most_expensive_hour=most_expensive_hour,
        solver_status="Greedy (PuLP unavailable)"
    )
