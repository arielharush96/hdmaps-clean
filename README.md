# HD-MAPS

Scaling Multi-Agent Coordination by Composition, Not Capacity.

Multi-agent reinforcement learning for autonomous-vehicle coordination almost universally fixes the fleet size at training time. We present hierarchical dynamic master-agent proto-plan system (HD-MAPS), a hierarchical deep reinforcement learning architecture that lifts this restriction through recursion rather than added capacity. The coordination signal emitted by a master network, which we call a proto-plan, is deliberately given the same size as the description of a single vehicle, making proto-plans and vehicle states interchangeable as inputs. One shared set of weights serves at every level of the hierarchies assembled at run time.

```
pip install -e .
python -m hdmaps train --pipeline
python -m hdmaps evaluate --protocol both
```
