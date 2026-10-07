from __future__ import annotations

from highway_env.road.lane import AbstractLane


def intersection_spacing(connector_length: float) -> float:
    lane_width = AbstractLane.DEFAULT_WIDTH
    right_turn_radius = lane_width + 5
    outer_distance = right_turn_radius + lane_width / 2
    return 2 * outer_distance + float(connector_length)


def zone_from_x(x: float, n_int: int, spacing: float) -> int:
    if n_int <= 1:
        return 0
    return max(0, min(n_int - 1, int(round(float(x) / spacing))))


def in_radius(position_xy, center_xy, radius: float) -> bool:
    dx = float(position_xy[0]) - float(center_xy[0])
    dy = float(position_xy[1]) - float(center_xy[1])
    return (dx * dx + dy * dy) ** 0.5 <= float(radius)
