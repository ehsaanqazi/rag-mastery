import { lazy, Suspense } from 'react'
import { Route, Routes } from 'react-router'
import Layout from './components/Layout'
import Home from './pages/Home'
import ModulePage from './pages/ModulePage'
import Roadmap from './pages/Roadmap'
import Library from './pages/Library'
import Projects from './pages/Projects'
import Assessments from './pages/Assessments'
import { PlaygroundIndex, PlaygroundPage } from './pages/Playground'
import NotFound from './pages/NotFound'

const LessonPage = lazy(() => import('./pages/LessonPage'))
const ProjectPage = lazy(() => import('./pages/ProjectPage'))
const ExamPage = lazy(() => import('./pages/ExamPage'))
const Progress = lazy(() => import('./pages/Progress'))

const later = (el: React.ReactNode) => <Suspense fallback={null}>{el}</Suspense>

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="roadmap" element={<Roadmap />} />
        <Route path="learn/:moduleSlug" element={<ModulePage />} />
        <Route path="learn/:moduleSlug/:topicSlug" element={later(<LessonPage />)} />
        <Route path="projects" element={<Projects />} />
        <Route path="projects/:slug" element={later(<ProjectPage />)} />
        <Route path="playground" element={<PlaygroundIndex />} />
        <Route path="playground/:slug" element={<PlaygroundPage />} />
        <Route path="assessments" element={<Assessments />} />
        <Route path="exam/:n" element={later(<ExamPage />)} />
        <Route path="library" element={<Library />} />
        <Route path="progress" element={later(<Progress />)} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}
