import React, { useState } from 'react'
import ReactDOM from 'react-dom/client'
import AutomatonDiagram from './demo.jsx'
import LocallyBalancedVisualizer from './golden_test_locally_balanced_bin_str.jsx'

const App = () => {
  const [currentPage, setCurrentPage] = useState('home')

  const pages = [
    {
      id: 'automaton',
      title: 'DP Automaton Diagram',
      description: 'Interactive state transition graph visualization',
      component: AutomatonDiagram,
      color: 'from-blue-500 to-purple-600'
    },
    {
      id: 'golden',
      title: 'Locally Balanced Checker',
      description: 'Animated window-by-window balance verification',
      component: LocallyBalancedVisualizer,
      color: 'from-indigo-500 to-blue-600'
    }
  ]

  // Home page with navigation
  if (currentPage === 'home') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-5xl font-bold text-center mb-4 text-white">
            M4 Verifier Visualizations
          </h1>
          <p className="text-center text-gray-300 mb-12 text-lg">
            Interactive tools for understanding locally balanced constraints
          </p>

          <div className="grid md:grid-cols-2 gap-6">
            {pages.map((page) => (
              <button
                key={page.id}
                onClick={() => setCurrentPage(page.id)}
                className={`bg-gradient-to-r ${page.color} p-8 rounded-2xl text-white text-left hover:scale-105 transition-transform shadow-2xl`}
              >
                <h2 className="text-2xl font-bold mb-2">{page.title}</h2>
                <p className="text-white/80">{page.description}</p>
                <div className="mt-4 text-sm font-semibold opacity-80">
                  Click to open →
                </div>
              </button>
            ))}
          </div>

          <div className="mt-12 text-center text-gray-400">
            <p className="text-sm">Part of the M4 Verifier project for Locally Balanced Constraints</p>
            <p className="text-xs mt-2">Based on &quot;Coding for Locally Balanced Constraints&quot; (Ge22)</p>
          </div>
        </div>
      </div>
    )
  }

  // Render selected page
  const selectedPage = pages.find(p => p.id === currentPage)
  const PageComponent = selectedPage?.component

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Navigation bar */}
      <nav className="bg-white shadow-md p-4 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <button
            onClick={() => setCurrentPage('home')}
            className="flex items-center gap-2 text-gray-700 hover:text-indigo-600 font-semibold"
          >
            ← Back to Home
          </button>
          <h1 className="text-xl font-bold text-gray-800">{selectedPage?.title}</h1>
          <div className="w-24"></div>
        </div>
      </nav>

      {/* Page content */}
      <main className="py-8">
        {PageComponent && <PageComponent />}
      </main>
    </div>
  )
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
