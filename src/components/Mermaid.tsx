import { useEffect, useId, useState } from 'react'
import { useTheme } from '../lib/store'

const palettes = {
  light: {
    primaryColor: '#ffffff',
    primaryTextColor: '#171a22',
    primaryBorderColor: '#4f46e5',
    lineColor: '#5b6172',
    secondaryColor: '#eaeafd',
    tertiaryColor: '#f0f2f6',
    background: '#f8f9fb',
    mainBkg: '#ffffff',
    clusterBkg: '#f0f2f6',
    clusterBorder: '#e1e4ec',
    edgeLabelBackground: '#f8f9fb',
    noteBkgColor: '#fbf0d9',
    noteTextColor: '#171a22',
    noteBorderColor: '#e1e4ec',
    actorBkg: '#ffffff',
    actorBorder: '#4f46e5',
    actorTextColor: '#171a22',
    signalColor: '#5b6172',
    signalTextColor: '#171a22',
    labelBoxBkgColor: '#eaeafd',
    labelBoxBorderColor: '#4f46e5',
    labelTextColor: '#171a22',
    loopTextColor: '#171a22',
    activationBkgColor: '#eaeafd',
    activationBorderColor: '#4f46e5',
    git0: '#4f46e5',
    git1: '#0e7490',
    git2: '#a15c07',
    git3: '#be185d',
    gitBranchLabel0: '#ffffff',
    gitBranchLabel1: '#ffffff',
    gitBranchLabel2: '#ffffff',
    gitBranchLabel3: '#ffffff',
    commitLabelColor: '#171a22',
    commitLabelBackground: '#f0f2f6',
  },
  dark: {
    primaryColor: '#151821',
    primaryTextColor: '#e8eaf0',
    primaryBorderColor: '#8f96ff',
    lineColor: '#9ca3b4',
    secondaryColor: '#1e2046',
    tertiaryColor: '#090a0f',
    background: '#0d0f15',
    mainBkg: '#151821',
    clusterBkg: '#11131a',
    clusterBorder: '#262a37',
    edgeLabelBackground: '#0d0f15',
    noteBkgColor: '#33270f',
    noteTextColor: '#e8eaf0',
    noteBorderColor: '#262a37',
    actorBkg: '#151821',
    actorBorder: '#8f96ff',
    actorTextColor: '#e8eaf0',
    signalColor: '#9ca3b4',
    signalTextColor: '#e8eaf0',
    labelBoxBkgColor: '#1e2046',
    labelBoxBorderColor: '#8f96ff',
    labelTextColor: '#e8eaf0',
    loopTextColor: '#e8eaf0',
    activationBkgColor: '#1e2046',
    activationBorderColor: '#8f96ff',
    git0: '#8f96ff',
    git1: '#5cc8dc',
    git2: '#e8b04b',
    git3: '#f28dbb',
    gitBranchLabel0: '#0d0f2b',
    gitBranchLabel1: '#06262d',
    gitBranchLabel2: '#2b1d05',
    gitBranchLabel3: '#330f22',
    commitLabelColor: '#e8eaf0',
    commitLabelBackground: '#151821',
  },
}

// mermaid keeps global state, so diagrams must render one at a time
let queue: Promise<unknown> = Promise.resolve()

function renderDiagram(id: string, code: string, theme: 'light' | 'dark'): Promise<string> {
  const job = queue.then(async () => {
    const { default: mermaid } = await import('mermaid')
    mermaid.initialize({
      startOnLoad: false,
      theme: 'base',
      look: 'classic',
      fontFamily: "'Inter Variable', system-ui, sans-serif",
      themeVariables: { ...palettes[theme], fontSize: '14px' },
    })
    const { svg } = await mermaid.render(id, code)
    return svg
  })
  queue = job.catch(() => {})
  return job
}

export default function Mermaid({ code }: { code: string }) {
  const theme = useTheme()
  const id = 'm' + useId().replace(/[^a-zA-Z0-9]/g, '')
  const [svg, setSvg] = useState('')
  const [error, setError] = useState(false)

  useEffect(() => {
    let cancelled = false
    renderDiagram(`${id}-${theme}`, code, theme)
      .then((svg) => {
        if (!cancelled) {
          setSvg(svg)
          setError(false)
        }
      })
      .catch((e) => {
        console.error('Mermaid render failed', e)
        if (!cancelled) setError(true)
      })
    return () => {
      cancelled = true
    }
  }, [code, theme, id])

  if (error)
    return (
      <pre data-mermaid-error className="overflow-x-auto rounded-xl border border-line bg-code p-4 font-mono text-xs text-muted">{code}</pre>
    )

  return (
    <figure
      className="mermaid-figure flex min-h-24 justify-center overflow-x-auto rounded-xl border border-line bg-surface px-4 py-6"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  )
}
