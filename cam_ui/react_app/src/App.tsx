import React, { useState, useEffect, useRef } from 'react'
import './App.css'
import Timeline, { Keyframes } from './Timeline'
import AppHeader from './AppHeader'
import AppBackground from './AppBackground'
import EditsPanel from './EditsPanel'

const MAX_FRAMES_LIMIT = 500  // Fallback when no data is loaded

// Viser runs on a different origin from the React app, so the iframe src can't go through the
// FastAPI `/api` proxy — the browser fetches it directly. Default is `http://localhost:VISER_PORT`,
// which works for local runs and SSH-forwarded runs. For a tunneled setup (pinggy etc.) where
// viser is reachable at a different URL than the React app, pass `?viser=<url>` in the page URL.
function resolveViserUrl(): string {
  const params = new URLSearchParams(window.location.search)
  const override = params.get('viser')
  if (override) return override
  return `http://localhost:${__VISER_PORT__}`
}

// Type definitions
interface HealthResponse {
  status: string
  viser_port?: number
}

interface ParsingInfo {
  depths_shape: string
  video_shape: string
  intrinsics_shape: string
  cameras_shape: string
  point_clouds_generated: number
}

interface CameraIntrinsics {
  fx: number
  fy: number
  cx: number
  cy: number
  matrix_3x3: number[][]
}

interface CameraExtrinsics {
  matrix_4x4: number[][]
  rotation_3x3: number[][]
  translation: number[]
  quaternion_wxyz: number[]
}

interface FirstFrameCamera {
  intrinsics: CameraIntrinsics
  extrinsics_c2w: CameraExtrinsics
  fov_degrees: number
  aspect_ratio: number
}

interface FileInfoResponse {
  status: string
  loaded: boolean
  file_path?: string
  num_frames?: number
  fps?: number
  parsing_info?: ParsingInfo
  first_frame_camera?: FirstFrameCamera
}

interface LoadFolderResponse {
  status: string
  message: string
  folder_name?: string
}

interface ErrorResponse {
  detail: string
}

interface FrameResponse {
  frame: number
  loaded: boolean
}

interface KeyframesResponse {
  keyframes: Keyframes
  count: number
}

interface CaptureKeyframeResponse {
  status: string
  frame: number
  pose: {
    position: number[]
    wxyz: number[]
    fov: number
    aspect: number
  }
}

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [filePath, setFilePath] = useState<string>('results/single/couple-newspaper/recon_and_seg')
  const [fileInfo, setFileInfo] = useState<FileInfoResponse | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [loadingMessage, setLoadingMessage] = useState<string>('')
  const [loadingFolder, setLoadingFolder] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  
  // Timeline state
  const [currentFrame, setCurrentFrame] = useState<number>(1)
  const [newKeyframes, setNewKeyframes] = useState<Keyframes>({})
  
  // Viewer aspect ratio
  const viewerWrapperRef = useRef<HTMLDivElement>(null)
  const [iframeStyle, setIframeStyle] = useState<React.CSSProperties>({ width: '100%', height: '100%' })

  // Export state
  const [outputFilename, setOutputFilename] = useState<string>('output_cameras.npz')
  const [exporting, setExporting] = useState<boolean>(false)
  const [exportSuccess, setExportSuccess] = useState<string | null>(null)

  // Edits panel readiness (backend builds unified buffers after load; PUT /api/edits needs them)
  const [editsReady, setEditsReady] = useState<boolean>(false)

  useEffect(() => {
    fetch('/api/health')
      .then(res => res.json())
      .then((data: HealthResponse) => setHealth(data))
      .catch(err => console.error('Failed to fetch health:', err))
  }, [])

  // Resize-aware aspect ratio for the Viser iframe
  useEffect(() => {
    const ratio = fileInfo?.first_frame_camera?.aspect_ratio
    const wrapper = viewerWrapperRef.current
    if (!ratio || !wrapper) {
      setIframeStyle({ width: '100%', height: '100%' })
      return
    }
    const compute = () => {
      const { clientWidth: w, clientHeight: h } = wrapper
      if (w <= 0 || h <= 0) return
      if (w / h > ratio) {
        setIframeStyle({ width: `${Math.floor(h * ratio)}px`, height: `${h}px` })
      } else {
        setIframeStyle({ width: `${w}px`, height: `${Math.floor(w / ratio)}px` })
      }
    }
    const observer = new ResizeObserver(compute)
    observer.observe(wrapper)
    compute()
    return () => observer.disconnect()
  }, [fileInfo?.first_frame_camera?.aspect_ratio])

  // Refs to avoid stale closures in poll callbacks
  const newKeyframesRef = useRef(newKeyframes)
  useEffect(() => { newKeyframesRef.current = newKeyframes }, [newKeyframes])
  const currentFrameRef = useRef(currentFrame)
  useEffect(() => { currentFrameRef.current = currentFrame }, [currentFrame])

  // Poll keyframes when file is loaded
  useEffect(() => {
    if (!fileInfo?.loaded) return

    const pollKeyframes = async () => {
      try {
        const response = await fetch('/api/keyframes')
        const data: KeyframesResponse = await response.json()

        // Update keyframes if they've changed
        if (JSON.stringify(data.keyframes) !== JSON.stringify(newKeyframesRef.current)) {
          setNewKeyframes(data.keyframes)
        }
      } catch (err) {
        console.error('Failed to poll keyframes:', err)
      }
    }

    // Poll every 1000ms
    const interval = setInterval(pollKeyframes, 1000)
    return () => clearInterval(interval)
  }, [fileInfo?.loaded])

  // Poll edits readiness — backend registers scene state on folder load, so we just need to
  // flip `editsReady` true once. DO NOT include `editsReady` in this effect's deps: re-including
  // it creates an infinite flip (setEditsReady(false) at top → poll → setEditsReady(true) →
  // effect re-runs → back to false), which propagates through as a rapid prop flip on
  // EditsPanel and keeps clearing its debounced PUT timer before the fetch can fire.
  useEffect(() => {
    if (!fileInfo?.loaded) {
      setEditsReady(false)
      return
    }
    setEditsReady(false)
    let stopped = false
    const pollReady = async () => {
      if (stopped) return
      try {
        const res = await fetch('/api/controls/state')
        const data = await res.json()
        if (stopped) return
        if (data.edits_ready) {
          setEditsReady(true)
          stopped = true
        }
      } catch (err) {
        console.error('Failed to poll edits readiness:', err)
      }
    }
    pollReady()
    const interval = setInterval(() => { if (!stopped) pollReady() }, 1500)
    return () => { stopped = true; clearInterval(interval) }
  }, [fileInfo?.loaded])

  // Poll current frame
  useEffect(() => {
    if (!fileInfo?.loaded) return

    const pollFrame = async () => {
      try {
        const response = await fetch('/api/frame/current')
        const data: FrameResponse = await response.json()

        if (data.frame !== currentFrameRef.current) {
          setCurrentFrame(data.frame)
        }
      } catch (err) {
        console.error('Failed to poll frame:', err)
      }
    }

    // Poll every 500ms
    const interval = setInterval(pollFrame, 500)
    return () => clearInterval(interval)
  }, [fileInfo?.loaded])

  const pollLoadStatus = async (): Promise<boolean> => {
    try {
      const response = await fetch('/api/load-status')
      const status: FileInfoResponse = await response.json()
      
      if (status.loaded) {
        setFileInfo(status)
        setLoading(false)
        setLoadingMessage('')
        setLoadingFolder(null)
        return true
      }
      return false
    } catch (err) {
      console.error('Failed to poll status:', err)
      return false
    }
  }

  const handleLoadFolder = async (): Promise<void> => {
    if (!filePath.trim()) {
      setError('Please enter a folder path')
      return
    }

    setLoading(true)
    setError(null)
    setFileInfo(null)
    setLoadingMessage('Resetting previous data ...')
    setLoadingFolder(null)

    // Reset timeline state
    setCurrentFrame(1)
    setNewKeyframes({})

    try {
      // First, trigger factory reset to clear any existing data
      try {
        await fetch('/api/factory-reset', { method: 'POST' })
        console.log('Factory reset completed before loading new folder')
      } catch (resetErr) {
        console.warn('Factory reset failed, continuing with load:', resetErr)
      }

      setLoadingMessage('Loading reconstruction folder ...')

      const response = await fetch('/api/load-folder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_path: filePath })
      })

      if (!response.ok) {
        const errorData: ErrorResponse = await response.json()
        throw new Error(errorData.detail || 'Failed to load folder')
      }

      const result: LoadFolderResponse = await response.json()
      setLoadingMessage(result.message || 'Loading reconstruction folder ...')
      if (result.folder_name) setLoadingFolder(result.folder_name)
      
      // Poll for completion
      const pollInterval = setInterval(async () => {
        const loaded = await pollLoadStatus()
        if (loaded) {
          clearInterval(pollInterval)
        }
      }, 1000)
      
      // Timeout after 2 minutes
      setTimeout(() => {
        clearInterval(pollInterval)
        if (loading) {
          setError('Loading timed out')
          setLoading(false)
        }
      }, 120000)
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred'
      setError(errorMessage)
      setLoading(false)
      setLoadingMessage('')
    }
  }
  
  const handleFrameClick = async (frame: number): Promise<void> => {
    setCurrentFrame(frame)
    
    // Send to backend to update Viser scene
    try {
      await fetch('/api/frame/set', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame })
      })
    } catch (err) {
      console.error('Failed to set frame:', err)
    }
  }
  
  const handleKeyframeToggle = async (frame: number): Promise<void> => {
    if (newKeyframes[frame]) {
      // Remove keyframe
      const updated = { ...newKeyframes }
      delete updated[frame]
      setNewKeyframes(updated)
      
      // Send to backend
      try {
        await fetch(`/api/keyframe/${frame}`, {
          method: 'DELETE'
        })
        console.log(`Removed keyframe at frame ${frame}`)
      } catch (err) {
        console.error('Failed to remove keyframe:', err)
      }
    } else {
      // Add keyframe - capture current camera pose
      try {
        const response = await fetch('/api/keyframe/capture', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ frame })
        })
        
        if (response.ok) {
          const data: CaptureKeyframeResponse = await response.json()
          setNewKeyframes({
            ...newKeyframes,
            [frame]: data.pose
          })
          console.log(`Added keyframe at frame ${frame}`)
        } else {
          console.error('Failed to capture keyframe')
        }
      } catch (err) {
        console.error('Error capturing keyframe:', err)
      }
    }
  }
  
  const handleExportTrajectory = async (): Promise<void> => {
    if (!fileInfo?.loaded) {
      setError('No data loaded')
      return
    }
    
    setExporting(true)
    setExportSuccess(null)
    
    try {
      const response = await fetch('/api/trajectory/export', { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: outputFilename })
      })
      
      if (response.ok) {
        const data = await response.json()
        const keyframeText = data.num_keyframes === 0 
          ? '(original cameras)'
          : `(${data.num_keyframes} keyframes)`
        setExportSuccess(`Exported ${data.num_frames} frames ${keyframeText}`)
      } else {
        setError('Failed to export cameras')
      }
    } catch (err) {
      console.error('Failed to export cameras:', err)
      setError('Failed to export cameras')
    } finally {
      setExporting(false)
    }
  }
  
  return (
    <AppBackground>
      <AppHeader
        title="Vista4D: Video Reshooting with 4D Point Clouds"
        showTooltip={false}
        badge="beta"
      />
      
      <main className="main">
        <aside className="sidebar">
          <div className="control-panel">
            <h2>Load reconstruction folder</h2>

            <div className="input-group">
              <label htmlFor="file-path">Folder path</label>
              <input
                id="file-path"
                type="text"
                value={filePath}
                onChange={(e) => setFilePath(e.target.value)}
                placeholder="/path/to/recon_and_seg/output"
                className="path-input"
              />
              <span className="input-hint">Output folder from recon_and_seg_single.py (contains video.mp4, depths/, cameras.npz)</span>
            </div>

            <button
              onClick={handleLoadFolder}
              disabled={loading}
              className="load-button"
            >
              {loading ? 'Loading ...' : 'Load reconstruction'}
            </button>

            {loading && loadingMessage && (
              <div className="loading-message">
                <div className="loading-spinner"></div>
                <p>
                  {loadingFolder
                    ? <>Loading <code className="path-text">{loadingFolder}</code> ...</>
                    : loadingMessage}
                </p>
                <p className="loading-subtext">Processing frames, please wait ...</p>
              </div>
            )}

            {error && (
              <div className="error-message">
                {error}
              </div>
            )}
            
            {/* Export Trajectory Section */}
            {fileInfo && fileInfo.loaded && (
              <div className="export-section">
                <h2>Export cameras</h2>
                
                <div className="input-group">
                  <label htmlFor="output-filename">Output filename</label>
                  <input
                    id="output-filename"
                    type="text"
                    value={outputFilename}
                    onChange={(e) => setOutputFilename(e.target.value)}
                    placeholder="output_cameras.npz"
                    className="path-input"
                  />
                  <span className="input-hint">Saved to cam_ui/exported_cameras/</span>
                </div>
                
                <button 
                  onClick={handleExportTrajectory} 
                  disabled={exporting || !fileInfo.loaded}
                  className="load-button"
                >
                  {exporting ? 'Exporting ...' : 'Export cameras'}
                </button>
                
                {exportSuccess && (
                  <div className="success-message">
                    {exportSuccess}
                  </div>
                )}
              </div>
            )}

            {fileInfo && fileInfo.loaded && (
              <div className="file-info">
                <h3>Loaded successfully</h3>

                <div className="info-section">
                  <h4>File details</h4>
                  <div className="info-item">
                    <strong>Path:</strong> 
                    <span className="path-text">{fileInfo.file_path}</span>
                  </div>
                  <div className="info-item">
                    <strong>Frames:</strong> {fileInfo.num_frames}
                  </div>
                  <div className="info-item">
                    <strong>FPS:</strong> {fileInfo.fps}
                  </div>
                </div>

                <div className="info-section">
                  <h4>Loaded data</h4>
                  <div className="parsing-info">
                    <div className="parse-item">
                      <strong>video:</strong> {fileInfo.parsing_info?.video_shape}
                      <span className="parse-desc">RGB frames from video.mp4</span>
                    </div>
                    <div className="parse-item">
                      <strong>depths:</strong> {fileInfo.parsing_info?.depths_shape}
                      <span className="parse-desc">Depth maps from depths/</span>
                    </div>
                    <div className="parse-item">
                      <strong>intrinsics:</strong> {fileInfo.parsing_info?.intrinsics_shape}
                      <span className="parse-desc">Camera parameters [fx, fy, cx, cy]</span>
                    </div>
                    <div className="parse-item">
                      <strong>cam_c2w:</strong> {fileInfo.parsing_info?.cameras_shape}
                      <span className="parse-desc">Camera-to-world transforms</span>
                    </div>
                  </div>
                </div>

                {fileInfo.first_frame_camera && (
                  <div className="info-section">
                    <h4>Frame 1 camera matrices</h4>
                    
                    <div className="matrix-section">
                      <h5>Intrinsics (K matrix)</h5>
                      <div className="matrix-params">
                        <span><strong>fx:</strong> {fileInfo.first_frame_camera.intrinsics.fx.toFixed(4)}</span>
                        <span><strong>fy:</strong> {fileInfo.first_frame_camera.intrinsics.fy.toFixed(4)}</span>
                        <span><strong>cx:</strong> {fileInfo.first_frame_camera.intrinsics.cx.toFixed(4)}</span>
                        <span><strong>cy:</strong> {fileInfo.first_frame_camera.intrinsics.cy.toFixed(4)}</span>
                      </div>
                      <div className="matrix-display">
                        {fileInfo.first_frame_camera.intrinsics.matrix_3x3.map((row, i) => (
                          <div key={i} className="matrix-row">
                            {row.map((val, j) => (
                              <span key={j} className="matrix-cell">
                                {val.toFixed(4)}
                              </span>
                            ))}
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="matrix-section">
                      <h5>Extrinsics (Camera-to-World)</h5>
                      <div className="matrix-params">
                        <span><strong>Translation:</strong> [{fileInfo.first_frame_camera.extrinsics_c2w.translation.map(v => v.toFixed(4)).join(', ')}]</span>
                      </div>
                      <div className="matrix-params">
                        <span><strong>Quaternion (wxyz):</strong> [{fileInfo.first_frame_camera.extrinsics_c2w.quaternion_wxyz.map(v => v.toFixed(4)).join(', ')}]</span>
                      </div>
                      <div className="matrix-display">
                        {fileInfo.first_frame_camera.extrinsics_c2w.matrix_4x4.map((row, i) => (
                          <div key={i} className="matrix-row">
                            {row.map((val, j) => (
                              <span key={j} className="matrix-cell">
                                {val.toFixed(4)}
                              </span>
                            ))}
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="matrix-params">
                      <span><strong>FOV:</strong> {fileInfo.first_frame_camera.fov_degrees.toFixed(2)}°</span>
                      <span><strong>Aspect:</strong> {fileInfo.first_frame_camera.aspect_ratio.toFixed(4)}</span>
                    </div>
                  </div>
                )}

                <div className="info-section">
                  <h4>Processing results</h4>
                  <div className="info-item">
                    <strong>Point clouds generated:</strong> {fileInfo.parsing_info?.point_clouds_generated}
                  </div>
                  <div className="success-badge">
                    Ready for visualization
                  </div>
                </div>
              </div>
            )}
          </div>
        </aside>

        <div className="viewer-container">
          <div className="viewer-wrapper" ref={viewerWrapperRef}>
            <iframe
              src={resolveViserUrl()}
              className="viser-frame"
              title="Viser 3D Viewer"
              style={iframeStyle}
            />
          </div>
          
          <Timeline
            currentFrame={currentFrame}
            onFrameClick={handleFrameClick}
            onKeyframeToggle={handleKeyframeToggle}
            newKeyframes={newKeyframes}
            isLoaded={fileInfo?.loaded || false}
            numFrames={fileInfo?.num_frames || MAX_FRAMES_LIMIT}
            videoFps={fileInfo?.fps}
          />
        </div>

        <EditsPanel
          isLoaded={fileInfo?.loaded || false}
          editsReady={editsReady}
        />
      </main>
    </AppBackground>
  )
}

export default App

