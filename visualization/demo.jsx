import React, { useState } from 'react';
import { ArrowRight, Code, Zap, Network } from 'lucide-react';

const AutomatonDiagram = () => {
  const [ell, setEll] = useState(4);
  const [delta, setDelta] = useState(1);
  const [highlightState, setHighlightState] = useState(null);

  // Calculate valid weight range
  const lo = Math.floor(ell / 2) - delta;
  const hi = Math.floor(ell / 2) + delta;
  
  // Generate all states for (ℓ-1) bits
  const numStates = Math.pow(2, ell - 1);
  const states = [];
  for (let i = 0; i < numStates; i++) {
    states.push(i.toString(2).padStart(ell - 1, '0'));
  }

  // Calculate weight of a bit string
  const weight = (bits) => bits.split('').filter(b => b === '1').length;

  // Build complete transition graph
  const buildGraph = () => {
    const graph = {};
    states.forEach(s => {
      graph[s] = { valid: [], invalid: [] };
      for (const bit of ['0', '1']) {
        const window = s + bit;
        const w = weight(window);
        const nextState = s.slice(1) + bit;
        if (w >= lo && w <= hi) {
          graph[s].valid.push({ bit, nextState, weight: w });
        } else {
          graph[s].invalid.push({ bit, nextState, weight: w });
        }
      }
    });
    return graph;
  };

  const graph = buildGraph();

  // Calculate positions for circular layout
  const getNodePosition = (index, total, radius = 200) => {
    const angle = (2 * Math.PI * index) / total - Math.PI / 2;
    return {
      x: 300 + radius * Math.cos(angle),
      y: 300 + radius * Math.sin(angle)
    };
  };

  // Count statistics
  const totalTransitions = states.reduce((sum, s) => sum + graph[s].valid.length, 0);
  const avgOutDegree = (totalTransitions / numStates).toFixed(2);

  // Small example for ℓ=4
  const SmallGraphDiagram = () => {
    if (numStates > 8) {
      return (
        <div className="text-center py-20 text-gray-500">
          <Network size={48} className="mx-auto mb-4 opacity-50" />
          <p className="text-lg font-semibold mb-2">Graph too large to display</p>
          <p className="text-sm">For ℓ={ell}, there are {numStates} states</p>
          <p className="text-sm">Use ℓ=4 to see the full diagram</p>
        </div>
      );
    }

    return (
      <svg width="600" height="600" className="mx-auto">
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
            fill="#10b981"
          >
            <polygon points="0 0, 10 3.5, 0 7" />
          </marker>
          <marker
            id="arrowhead-red"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
            fill="#ef4444"
          >
            <polygon points="0 0, 10 3.5, 0 7" />
          </marker>
          <marker
            id="arrowhead-highlight"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
            fill="#eab308"
          >
            <polygon points="0 0, 10 3.5, 0 7" />
          </marker>
        </defs>

        {/* Draw edges first (behind nodes) */}
        {states.map((state, idx) => {
          const pos = getNodePosition(idx, numStates);
          const transitions = graph[state];
          
          return (
            <g key={`edges-${state}`}>
              {/* Valid transitions */}
              {transitions.valid.map((trans, tIdx) => {
                const targetIdx = states.indexOf(trans.nextState);
                const targetPos = getNodePosition(targetIdx, numStates);
                
                // Calculate control point for curved edge
                const midX = (pos.x + targetPos.x) / 2;
                const midY = (pos.y + targetPos.y) / 2;
                const dx = targetPos.x - pos.x;
                const dy = targetPos.y - pos.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                const offset = Math.min(30, dist * 0.2);
                const ctrlX = midX - (dy / dist) * offset;
                const ctrlY = midY + (dx / dist) * offset;
                
                const isHighlighted = highlightState === state;
                
                return (
                  <g key={`edge-${state}-${trans.bit}-${tIdx}`}>
                    <path
                      d={`M ${pos.x} ${pos.y} Q ${ctrlX} ${ctrlY} ${targetPos.x} ${targetPos.y}`}
                      fill="none"
                      stroke={isHighlighted ? "#eab308" : "#10b981"}
                      strokeWidth={isHighlighted ? "3" : "2"}
                      opacity={isHighlighted ? "1" : "0.4"}
                      markerEnd={isHighlighted ? "url(#arrowhead-highlight)" : "url(#arrowhead)"}
                    />
                    {isHighlighted && (
                      <text
                        x={ctrlX}
                        y={ctrlY}
                        fill="#eab308"
                        fontSize="14"
                        fontWeight="bold"
                        textAnchor="middle"
                      >
                        {trans.bit}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          );
        })}

        {/* Draw nodes on top */}
        {states.map((state, idx) => {
          const pos = getNodePosition(idx, numStates);
          const isHighlighted = highlightState === state;
          
          return (
            <g 
              key={`node-${state}`}
              onMouseEnter={() => setHighlightState(state)}
              onMouseLeave={() => setHighlightState(null)}
              style={{ cursor: 'pointer' }}
            >
              <circle
                cx={pos.x}
                cy={pos.y}
                r={isHighlighted ? "35" : "30"}
                fill={isHighlighted ? "#fef08a" : "#dbeafe"}
                stroke={isHighlighted ? "#eab308" : "#3b82f6"}
                strokeWidth={isHighlighted ? "4" : "3"}
              />
              <text
                x={pos.x}
                y={pos.y + 5}
                textAnchor="middle"
                fontSize="16"
                fontWeight="bold"
                fontFamily="monospace"
                fill={isHighlighted ? "#92400e" : "#1e40af"}
              >
                {state}
              </text>
            </g>
          );
        })}
      </svg>
    );
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="bg-white rounded-xl shadow-2xl p-8">
        <h1 className="text-3xl font-bold mb-2 text-center bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          DP Automaton: State Transition Graph
        </h1>
        <p className="text-center text-gray-600 mb-8">Interactive visualization of the automaton structure</p>

        {/* Parameters */}
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 p-6 rounded-xl border-2 border-indigo-200 mb-8">
          <h3 className="text-xl font-bold text-indigo-900 mb-4">🎛️ Automaton Parameters</h3>
          <div className="grid md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-semibold mb-2 text-gray-700">
                Window Length (ℓ): {ell}
              </label>
              <input 
                type="range" 
                min="4" 
                max="6" 
                step="2"
                value={ell}
                onChange={(e) => setEll(Number(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-600 mt-1">
                States: {numStates} = 2^{ell-1}
              </div>
            </div>
            <div>
              <label className="block text-sm font-semibold mb-2 text-gray-700">
                Delta (δ): {delta}
              </label>
              <input 
                type="range" 
                min="1" 
                max="2"
                value={delta}
                onChange={(e) => setDelta(Number(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-600 mt-1">
                Valid weight: [{lo}, {hi}]
              </div>
            </div>
            <div className="bg-white p-4 rounded-lg border-2 border-indigo-200">
              <div className="text-xs text-gray-600">Graph Stats</div>
              <div className="text-lg font-bold text-indigo-700">
                {totalTransitions} edges
              </div>
              <div className="text-xs text-gray-600">
                Avg out-degree: {avgOutDegree}
              </div>
            </div>
          </div>
        </div>

        {/* Main State Transition Diagram */}
        <div className="bg-gradient-to-br from-blue-50 to-cyan-50 p-6 rounded-xl border-2 border-blue-300 mb-8">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-2xl font-bold text-blue-900 flex items-center gap-2">
              <Network size={28} />
              State Transition Graph
            </h3>
            <div className="text-sm text-gray-700 bg-white px-3 py-1 rounded-lg">
              Hover over nodes to highlight transitions
            </div>
          </div>

          <div className="bg-white rounded-xl p-4 mb-4">
            <SmallGraphDiagram />
          </div>

          {highlightState && (
            <div className="bg-yellow-50 p-4 rounded-lg border-2 border-yellow-300">
              <h4 className="font-bold text-yellow-900 mb-2">
                State "{highlightState}" Transitions:
              </h4>
              <div className="grid md:grid-cols-2 gap-3 text-sm">
                {graph[highlightState].valid.map((trans, idx) => (
                  <div key={idx} className="bg-white p-3 rounded-lg border border-green-300">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-mono font-bold">{highlightState}</span>
                      <span className="text-gray-400">+</span>
                      <span className="font-mono font-bold text-blue-600">{trans.bit}</span>
                      <ArrowRight size={16} className="text-green-600" />
                      <span className="font-mono font-bold text-purple-600">{trans.nextState}</span>
                    </div>
                    <div className="text-xs text-gray-600">
                      Window: {highlightState}{trans.bit} (weight: {trans.weight}) ✓
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Legend */}
        <div className="bg-white p-6 rounded-xl border-2 border-gray-200 mb-8">
          <h3 className="text-lg font-bold text-gray-800 mb-4">📖 Graph Legend</h3>
          <div className="grid md:grid-cols-3 gap-4 text-sm">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-blue-200 border-2 border-blue-600 flex-shrink-0"></div>
              <div>
                <div className="font-semibold">Nodes (States)</div>
                <div className="text-gray-600 text-xs">All {ell-1}-bit strings ({numStates} total)</div>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0">
                <svg width="32" height="20">
                  <line x1="0" y1="10" x2="32" y2="10" stroke="#10b981" strokeWidth="2" markerEnd="url(#arrowhead)" />
                </svg>
              </div>
              <div>
                <div className="font-semibold">Valid Transitions</div>
                <div className="text-gray-600 text-xs">Window weight ∈ [{lo}, {hi}]</div>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-yellow-100 border-2 border-yellow-500 flex-shrink-0"></div>
              <div>
                <div className="font-semibold">Highlighted State</div>
                <div className="text-gray-600 text-xs">Shows outgoing edges</div>
              </div>
            </div>
          </div>
        </div>

        {/* Transition Table */}
        <div className="bg-gradient-to-br from-purple-50 to-pink-50 p-6 rounded-xl border-2 border-purple-300 mb-8">
          <h3 className="text-xl font-bold text-purple-900 mb-4">📋 Complete Transition Table</h3>
          <div className="bg-white rounded-lg overflow-hidden border-2 border-purple-200">
            <table className="w-full text-sm">
              <thead className="bg-purple-100">
                <tr>
                  <th className="px-4 py-3 text-left font-semibold">State</th>
                  <th className="px-4 py-3 text-left font-semibold">+0 → Next</th>
                  <th className="px-4 py-3 text-left font-semibold">+1 → Next</th>
                  <th className="px-4 py-3 text-left font-semibold">Out-degree</th>
                </tr>
              </thead>
              <tbody>
                {states.map((state, idx) => (
                  <tr 
                    key={state}
                    className={`border-t hover:bg-purple-50 ${idx % 2 === 0 ? 'bg-gray-50' : ''}`}
                    onMouseEnter={() => setHighlightState(state)}
                    onMouseLeave={() => setHighlightState(null)}
                  >
                    <td className="px-4 py-3 font-mono font-bold text-blue-700">{state}</td>
                    <td className="px-4 py-3 font-mono">
                      {(() => {
                        const trans = graph[state].valid.find(t => t.bit === '0');
                        return trans ? (
                          <span className="text-green-700">
                            {trans.nextState} <span className="text-xs text-gray-500">(w={trans.weight})</span>
                          </span>
                        ) : (
                          <span className="text-red-500">✗</span>
                        );
                      })()}
                    </td>
                    <td className="px-4 py-3 font-mono">
                      {(() => {
                        const trans = graph[state].valid.find(t => t.bit === '1');
                        return trans ? (
                          <span className="text-green-700">
                            {trans.nextState} <span className="text-xs text-gray-500">(w={trans.weight})</span>
                          </span>
                        ) : (
                          <span className="text-red-500">✗</span>
                        );
                      })()}
                    </td>
                    <td className="px-4 py-3">
                      <span className="font-bold text-purple-700">{graph[state].valid.length}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Code Implementation */}
        <div className="bg-gray-900 text-gray-100 p-6 rounded-xl">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Code size={24} />
            Python Code: Building the Graph
          </h3>
          <pre className="text-sm overflow-x-auto">
{`def build_transition_graph(ell, delta):
    """Build state transition graph for DP automaton."""
    
    # Step 1: Generate all (ℓ-1)-bit states
    state_len = ell - 1
    num_states = 2 ** state_len
    states = [format(i, f'0{state_len}b') for i in range(num_states)]
    
    # Step 2: Calculate valid weight range
    lo = ell // 2 - delta
    hi = ell // 2 + delta
    
    # Step 3: Build adjacency list
    graph = {s: [] for s in states}
    
    for state in states:
        for bit in ['0', '1']:
            # Form ℓ-bit window
            window = state + bit
            weight = sum(1 for c in window if c == '1')
            
            # Check if window is valid
            if lo <= weight <= hi:
                next_state = state[1:] + bit  # Shift left
                graph[state].append(next_state)
    
    return states, graph

# Example: ℓ=${ell}, δ=${delta}
states, graph = build_transition_graph(${ell}, ${delta})

print(f"States: {len(states)}")  # ${numStates}
print(f"Transitions: {sum(len(v) for v in graph.values())}")  # ${totalTransitions}

# Example transitions from state "${states[0]}":
print(graph[\"${states[0]}\"])  # Example output`}
          </pre>
        </div>

        {/* Key Insights */}
        <div className="mt-8 bg-gradient-to-r from-green-50 to-emerald-50 p-6 rounded-xl border-2 border-green-300">
          <h3 className="text-xl font-bold text-green-900 mb-4 flex items-center gap-2">
            <Zap size={24} />
            Key Insights from the Graph
          </h3>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div className="bg-white p-4 rounded-lg">
              <div className="font-bold text-green-800 mb-2">🎯 Graph Structure</div>
              <ul className="space-y-1 text-gray-700">
                <li>• <strong>{numStates} nodes</strong> (states)</li>
                <li>• <strong>{totalTransitions} directed edges</strong> (valid transitions)</li>
                <li>• Average out-degree: <strong>{avgOutDegree}</strong></li>
                <li>• Each node has at most 2 outgoing edges (+0, +1)</li>
              </ul>
            </div>
            <div className="bg-white p-4 rounded-lg">
              <div className="font-bold text-green-800 mb-2">⚡ Why This Works</div>
              <ul className="space-y-1 text-gray-700">
                <li>• Graph is <strong>fixed size</strong> (doesn't grow with n)</li>
                <li>• DP "walks" through graph for n steps</li>
                <li>• Counts all valid paths from any start to any end</li>
                <li>• Complexity: <strong>O(n × {numStates})</strong> instead of O(2^n)</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AutomatonDiagram;