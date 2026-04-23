import React, { useState, useMemo, useEffect } from 'react'
import './Timeline.css'

const MAX_FRAMES_LIMIT = 500  // Hard maximum limit (matches backend MAX_FRAMES)

// Type definitions
interface CameraPose {
  position: number[]
  wxyz: number[]
  fov: number
  aspect: number
}

interface Keyframes {
  [frame: number]: CameraPose
}

interface TimelineProps {
  currentFrame: number
  onFrameClick: (frame: number) => void
  onKeyframeToggle: (frame: number) => void
  newKeyframes: Keyframes
  isLoaded: boolean
  numFrames: number  // Actual frame count from loaded NPZ
  videoFps?: number  // FPS of the loaded video (used as default playback rate)
}

interface ControlsState {
  loaded: boolean
  current_frame: number
  num_frames: number
  follow_camera: boolean
  captured_count: number
  status: string
  trajectory_tension: number
  point_size: number
  show_static_bg: boolean
  has_static_bg: boolean
  show_source_cameras: boolean
  has_dse: boolean
  num_dse_frames: number
  dse_frame_interval: number
  show_dse_cameras: boolean
}

interface Frame1Intrinsics {
  fx: number
  fy: number
  cx: number
  cy: number
  fov: number
  fov_vertical: number
  aspect: number
  width: number
  height: number
}

interface IntrinsicsOverride {
  fx: number
  fy: number
  cx: number
  cy: number
}

interface Segment {
  a: number
  b: number
  kind: 'OldNew' | 'NewOld' | 'NewNew'
}

interface Breakpoint {
  frame: number
  isNew: boolean
}

interface InterpolationInfo {
  kind: 'OldNew' | 'NewOld' | 'NewNew'
  progress: number
  segment: Segment
}

type RowType = 'original' | 'new' | 'interpolated'

/**
 * Camera Timeline Component
 * 
 * Displays 3 rows of frame pills (dynamic count based on loaded NPZ):
 * - Top (New): User-defined keyframes (green when set)
 * - Middle (Interpolated): Interpolation segments (white→purple or green→purple)
 * - Bottom (Original): Original trajectory (all white, read-only)
 */
function Timeline({ currentFrame, onFrameClick, onKeyframeToggle, newKeyframes, isLoaded, numFrames, videoFps }: TimelineProps) {
  const [hoveredFrame, setHoveredFrame] = useState<number | null>(null)
  const [hoveredRow, setHoveredRow] = useState<RowType | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  
  // Control panel state
  const [showStaticBg, setShowStaticBg] = useState(false)
  const [hasStaticBg, setHasStaticBg] = useState(false)
  const [showSourceCameras, setShowSourceCameras] = useState(false)
  const [hasDse, setHasDse] = useState(false)
  const [numDseFrames, setNumDseFrames] = useState(0)
  const [showDseCameras, setShowDseCameras] = useState(false)
  const [dseFrameInterval, setDseFrameInterval] = useState(1)
  const [localDseFrameInterval, setLocalDseFrameInterval] = useState(1)
  const [dseFrameIntervalDirty, setDseFrameIntervalDirty] = useState(false)
  const [updatingDseFrameInterval, setUpdatingDseFrameInterval] = useState(false)
  const [followCamera, setFollowCamera] = useState(false)
  const [capturedCount, setCapturedCount] = useState(0)
  const [trajectoryTension, setTrajectoryTension] = useState(0.5)
  const [localTension, setLocalTension] = useState(0.5)
  const [tensionDirty, setTensionDirty] = useState(false)
  const [recomputing, setRecomputing] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playbackFps, setPlaybackFps] = useState(videoFps ?? 12)
  const [pointSize, setPointSize] = useState(0.003)
  const [localPointSize, setLocalPointSize] = useState(0.003)
  const [pointSizeDirty, setPointSizeDirty] = useState(false)
  const [updatingPointSize, setUpdatingPointSize] = useState(false)
  
  // Intrinsics control state
  const [frame1Intrinsics, setFrame1Intrinsics] = useState<Frame1Intrinsics | null>(null)
  const [showIntrinsicsDropdown, setShowIntrinsicsDropdown] = useState(false)
  const [fovMultiplier, setFovMultiplier] = useState(1.0)
  const [originalViewportFov, setOriginalViewportFov] = useState<number | null>(null)

  // Sync playback FPS when video FPS becomes available (arrives after load)
  useEffect(() => {
    if (videoFps != null && !isPlaying) {
      setPlaybackFps(Math.min(videoFps, 12))
    }
  }, [videoFps])

  // Compute interpolation segments based on new keyframes
  const segments = useMemo(() => {
    return computeSegments(newKeyframes, numFrames)
  }, [newKeyframes, numFrames])
  
  // Fetch frame 1 intrinsics as reference baseline
  useEffect(() => {
    if (!isLoaded) return

    const fetchFrame1Intrinsics = async () => {
      try {
        const response = await fetch('/api/load-status')
        const data = await response.json()
        if (data.first_frame_camera) {
          setFrame1Intrinsics({
            fx: data.first_frame_camera.intrinsics.fx,
            fy: data.first_frame_camera.intrinsics.fy,
            cx: data.first_frame_camera.intrinsics.cx,
            cy: data.first_frame_camera.intrinsics.cy,
            fov: data.first_frame_camera.fov_degrees,
            fov_vertical: data.first_frame_camera.fov_vertical_degrees,
            aspect: data.first_frame_camera.aspect_ratio,
            width: 1280,
            height: 720
          })
        }
      } catch (err) {
        console.error('Failed to fetch frame 1 intrinsics:', err)
      }
    }

    fetchFrame1Intrinsics()
  }, [isLoaded])
  
  // Poll controls state
  useEffect(() => {
    if (!isLoaded) return

    const pollControls = async () => {
      try {
        const response = await fetch('/api/controls/state')
        const data: ControlsState = await response.json()
        
        setShowStaticBg(data.show_static_bg)
        setHasStaticBg(data.has_static_bg)
        setShowSourceCameras(data.show_source_cameras)
        setFollowCamera(data.follow_camera)
        setCapturedCount(data.captured_count)

        setHasDse(data.has_dse)
        setNumDseFrames(data.num_dse_frames)
        setShowDseCameras(data.show_dse_cameras)
        setDseFrameInterval(data.dse_frame_interval)
        if (data.dse_frame_interval === localDseFrameInterval) {
          setDseFrameIntervalDirty(false)
        }

        // Update backend tension value
        setTrajectoryTension(data.trajectory_tension)

        // If backend tension matches local, mark as clean
        if (Math.abs(data.trajectory_tension - localTension) < 0.001) {
          setTensionDirty(false)
        }

        // Update backend point size value
        setPointSize(data.point_size)

        // If backend point size matches local, mark as clean
        if (Math.abs(data.point_size - localPointSize) < 0.0001) {
          setPointSizeDirty(false)
        }
      } catch (err) {
        console.error('Failed to poll controls state:', err)
      }
    }

    const interval = setInterval(pollControls, 1000)
    pollControls()

    return () => clearInterval(interval)
  }, [isLoaded, localTension, localPointSize, localDseFrameInterval])

  // (Playback is server-side; no client-side loop needed)

  // Drag handling for scrubber
  const calculateFrameFromPosition = (clientX: number, element: HTMLElement): number => {
    const rect = element.getBoundingClientRect()
    const x = clientX - rect.left
    // Use scrollWidth (full content width including overflow) so the calculation
    // is correct whether the timeline fits the screen or overflows and scrolls.
    const totalWidth = element.scrollWidth
    const percentage = Math.max(0, Math.min(1, x / totalWidth))
    return Math.max(1, Math.min(numFrames, Math.round(percentage * (numFrames - 1)) + 1))
  }

  const handleScrubberMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    setIsDragging(true)
    const frame = calculateFrameFromPosition(e.clientX, e.currentTarget)
    onFrameClick(frame)
  }

  React.useEffect(() => {
    if (!isDragging) return

    const handleMouseMove = (e: MouseEvent) => {
      const scrubber = document.querySelector('.timeline-scrubber') as HTMLElement
      if (!scrubber) return
      const frame = calculateFrameFromPosition(e.clientX, scrubber)
      onFrameClick(frame)
    }

    const handleMouseUp = () => {
      setIsDragging(false)
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isDragging, onFrameClick])

  // Get interpolation info for a specific frame
  const getInterpolationInfo = (frame: number): InterpolationInfo | null => {
    for (const segment of segments) {
      if (frame > segment.a && frame < segment.b) {
        // Frame is in interior of segment
        const progress = (frame - segment.a) / (segment.b - segment.a)
        return { kind: segment.kind, progress, segment }
      }
    }
    return null
  }

  const handlePillClick = async (frame: number, row: RowType): Promise<void> => {
    if (row === 'new') {
      // If it's a keyframe, delete it directly
      const isKeyframe = newKeyframes[frame] !== undefined
      if (isKeyframe) {
        try {
          const response = await fetch(`/api/keyframe/${frame}`, {
            method: 'DELETE'
          })
          
          if (response.ok) {
            console.log(`Deleted keyframe at frame ${frame}`)
          } else {
            console.error('Failed to delete keyframe')
          }
        } catch (err) {
          console.error('Error deleting keyframe:', err)
        }
      }
    } else {
      // Move playhead to this frame
      onFrameClick(frame)
    }
  }

  const renderPill = (frame: number, row: RowType) => {
    const isCurrentFrame = frame === currentFrame
    const isHovered = hoveredFrame === frame && hoveredRow === row
    const isKeyframe = newKeyframes[frame] !== undefined
    
    let pillClass = 'timeline-pill'
    let style: React.CSSProperties = {}
    let title = `Frame ${frame}`

    if (row === 'original') {
      // Original row: all white
      pillClass += ' pill-original'
      
      // Dim if there's a new keyframe at this frame OR if this frame is interpolated
      const interpInfo = getInterpolationInfo(frame)
      if (isKeyframe) {
        pillClass += ' pill-dimmed'
        title += ' (Original - overridden by keyframe)'
      } else if (interpInfo) {
        pillClass += ' pill-dimmed'
        title += ' (Original - interpolated, not used)'
      } else {
        title += ' (Original)'
      }
      
    } else if (row === 'new') {
      // New row: green if keyframe exists (read-only)
      pillClass += ' pill-new'
      if (isKeyframe) {
        pillClass += ' pill-keyframe'
        title += ' (Keyframe - click to delete)'
      } else {
        pillClass += ' pill-empty'
        title += ' (Use Viser "Capture Current View" to add keyframe)'
      }
    } else if (row === 'interpolated') {
      // Interpolated row: show interpolation segments with staircase fill
      const interpInfo = getInterpolationInfo(frame)
      if (interpInfo) {
        pillClass += ' pill-interpolated'
        const fillPercent = interpInfo.progress * 100
        
        if (interpInfo.kind === 'OldNew') {
          pillClass += ' pill-interp-old-new'
          title += ` (Interpolated: Old→New, ${Math.round(fillPercent)}%)`
          // Staircase: fill from bottom with accent color, white background
          style.background = `linear-gradient(to top, #F1A2FF ${fillPercent}%, #FFFFFF ${fillPercent}%)`
        } else if (interpInfo.kind === 'NewOld') {
          pillClass += ' pill-interp-new-old'
          title += ` (Interpolated: New→Old, ${Math.round(fillPercent)}%)`
          // Staircase: white grows from top down, accent stays at bottom
          style.background = `linear-gradient(to bottom, #FFFFFF ${fillPercent}%, #F1A2FF ${fillPercent}%)`
        } else {
          pillClass += ' pill-interp-new-new'
          title += ` (Interpolated: New→New, ${Math.round(fillPercent)}%)`
          // Staircase: fill from bottom with accent color, accent background
          style.background = `linear-gradient(to top, #F1A2FF ${fillPercent}%, #F1A2FF ${fillPercent}%)`
        }
      } else {
        pillClass += ' pill-empty'
        title += ' (No interpolation)'
      }
    }

    if (isHovered) pillClass += ' pill-hovered'

    return (
      <div
        key={frame}
        className={pillClass}
        style={style}
        title={title}
        onClick={() => handlePillClick(frame, row)}
        onMouseEnter={() => {
          setHoveredFrame(frame)
          setHoveredRow(row)
        }}
        onMouseLeave={() => {
          setHoveredFrame(null)
          setHoveredRow(null)
        }}
      >
        <span className="pill-frame-number">{frame}</span>
      </div>
    )
  }

  // Control handlers
  const handleStaticBgToggle = async (show: boolean) => {
    setShowStaticBg(show)
    try {
      await fetch('/api/static-bg/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ show })
      })
    } catch (err) {
      console.error('Failed to toggle static background:', err)
      setShowStaticBg(!show)
    }
  }

  const handleSourceCamerasToggle = async (show: boolean) => {
    setShowSourceCameras(show)
    try {
      await fetch('/api/source-cameras/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ show })
      })
    } catch (err) {
      console.error('Failed to toggle source cameras:', err)
      setShowSourceCameras(!show)
    }
  }

  const handleDseCamerasToggle = async (show: boolean) => {
    setShowDseCameras(show)
    try {
      await fetch('/api/dse-cameras/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ show })
      })
    } catch (err) {
      console.error('Failed to toggle DSE cameras:', err)
      setShowDseCameras(!show)
    }
  }

  const handleDseFrameIntervalChange = (value: number) => {
    setLocalDseFrameInterval(value)
    setDseFrameIntervalDirty(value !== dseFrameInterval)
  }

  const handleUpdateDseFrameInterval = async () => {
    setUpdatingDseFrameInterval(true)
    try {
      await fetch('/api/dse/frame-interval', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame_interval: localDseFrameInterval })
      })
      setDseFrameIntervalDirty(false)
    } catch (err) {
      console.error('Failed to update DSE frame interval:', err)
    } finally {
      setUpdatingDseFrameInterval(false)
    }
  }

  const handleFollowCamera = async () => {
    const newValue = !followCamera
    setFollowCamera(newValue)
    try {
      await fetch('/api/cameras/follow', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ follow: newValue })
      })
    } catch (err) {
      console.error('Failed to toggle follow camera:', err)
      setFollowCamera(!newValue)
    }
  }

  const handleSnapCamera = async () => {
    try {
      await fetch('/api/cameras/snap', { method: 'POST' })
    } catch (err) {
      console.error('Failed to snap camera:', err)
    }
  }


  const handleFactoryReset = async () => {
    if (!window.confirm('Factory reset will clear ALL data. Continue?')) return
    try {
      await fetch('/api/playback/stop', { method: 'POST' })
      setIsPlaying(false)
      await fetch('/api/factory-reset', { method: 'POST' })
    } catch (err) {
      console.error('Failed to factory reset:', err)
      alert('Failed to factory reset')
    }
  }

  const handleResetKeyframes = async () => {
    if (!window.confirm('Reset keyframes will clear all user-defined camera poses and return to the original trajectory. Continue?')) return
    try {
      await fetch('/api/keyframes/clear', { method: 'POST' })
      console.log('Reset keyframes - cleared all user-defined poses')
    } catch (err) {
      console.error('Failed to reset keyframes:', err)
      alert('Failed to reset keyframes')
    }
  }

  const handleTensionChange = (value: number) => {
    setLocalTension(value)
    // Mark as dirty if different from backend value
    if (Math.abs(value - trajectoryTension) > 0.001) {
      setTensionDirty(true)
    } else {
      setTensionDirty(false)
    }
  }

  const handleRecomputeTension = async () => {
    setRecomputing(true)
    try {
      await fetch('/api/trajectory/tension', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tension: localTension })
      })
      console.log(`Recomputing trajectory with tension ${localTension}`)
      setTensionDirty(false)
    } catch (err) {
      console.error('Failed to update tension:', err)
    } finally {
      setRecomputing(false)
    }
  }

  const handlePlayPause = async () => {
    const newPlaying = !isPlaying
    setIsPlaying(newPlaying)
    try {
      if (newPlaying) {
        await fetch('/api/playback/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fps: playbackFps })
        })
      } else {
        await fetch('/api/playback/stop', { method: 'POST' })
      }
    } catch (err) {
      console.error('Failed to toggle playback:', err)
      setIsPlaying(!newPlaying)  // Revert on error
    }
  }

  const handleFpsChange = async (fps: number) => {
    setPlaybackFps(fps)
    if (isPlaying) {
      try {
        await fetch('/api/playback/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ fps })
        })
      } catch (err) {
        console.error('Failed to update playback fps:', err)
      }
    }
  }

  const handlePointSizeChange = (value: number) => {
    setLocalPointSize(value)
    // Mark as dirty if different from backend value
    if (Math.abs(value - pointSize) > 0.0001) {
      setPointSizeDirty(true)
    } else {
      setPointSizeDirty(false)
    }
  }

  const handleUpdatePointSize = async () => {
    setUpdatingPointSize(true)
    try {
      const response = await fetch('/api/pointcloud/size', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ point_size: localPointSize })
      })
      const data = await response.json()
      console.log(`Updating point size to ${data.point_size}`)
      setPointSize(data.point_size)
      setPointSizeDirty(false)
    } catch (err) {
      console.error('Failed to update point size:', err)
    } finally {
      setUpdatingPointSize(false)
    }
  }

  // Calculate intrinsics from FOV multiplier
  const calculateIntrinsicsFromFov = (multiplier: number): IntrinsicsOverride | null => {
    if (!frame1Intrinsics) return null
    

    const baseFAvg = (frame1Intrinsics.fx + frame1Intrinsics.fy) / 2.0
    

    const newFAvg = baseFAvg * multiplier
    
    // Set fx = fy = f_avg (assuming square pixels, matching backend)
    const newFx = newFAvg
    const newFy = newFAvg
    
    return {
      fx: newFx,
      fy: newFy,
      cx: frame1Intrinsics.cx,
      cy: frame1Intrinsics.cy
    }
  }

  const handleToggleCaptureDropdown = async () => {
    if (!showIntrinsicsDropdown) {
      // Opening dropdown - save current viewport FOV so we can revert on cancel
      try {
        const response = await fetch('/api/viewport/fov')
        const data = await response.json()
        setOriginalViewportFov(data.fov_multiplier || 1.0)
      } catch (err) {
        console.error('Failed to get viewport FOV:', err)
        setOriginalViewportFov(1.0)
      }
      // Initialize zoom from existing keyframe (if any), otherwise 1.0x.
      // Use focal lengths (not FOV angles) — FOV is nonlinear so the ratio is wrong.
      // All zooms are relative to frame 1's average focal length.
      const existingKeyframe = newKeyframes[currentFrame] as any
      const frame1FAvg = frame1Intrinsics ? (frame1Intrinsics.fx + frame1Intrinsics.fy) / 2.0 : 1.0
      const initialMultiplier = (existingKeyframe?.fx && frame1FAvg > 0)
        ? existingKeyframe.fx / frame1FAvg
        : 1.0
      setFovMultiplier(initialMultiplier)
      await updateViewportFov(initialMultiplier)
      setShowIntrinsicsDropdown(true)
    } else {
      // Closing dropdown - revert viewport FOV
      if (originalViewportFov !== null) {
        await updateViewportFov(originalViewportFov)
      }
      setShowIntrinsicsDropdown(false)
    }
  }

  const updateViewportFov = async (multiplier: number) => {
    if (!frame1Intrinsics) return
    
    try {
      await fetch('/api/viewport/fov', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fov_multiplier: multiplier })
      })
    } catch (err) {
      console.error('Failed to update viewport FOV:', err)
    }
  }

  const handleFovMultiplierChange = async (value: number) => {
    setFovMultiplier(value)
    // Update viewport FOV in real-time
    await updateViewportFov(value)
  }

  const handleConfirmCapture = async () => {
    if (!frame1Intrinsics) return
    
    try {
      const intrinsicsOverride = calculateIntrinsicsFromFov(fovMultiplier)

      const response = await fetch('/api/keyframe/capture', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frame: currentFrame,
          intrinsics_override: intrinsicsOverride
        })
      })

      if (response.ok) {
        const data = await response.json()
        console.log('Captured keyframe with custom intrinsics:', data)
        setShowIntrinsicsDropdown(false)
      } else {
        console.error('Failed to capture keyframe')
      }
    } catch (err) {
      console.error('Failed to capture with custom intrinsics:', err)
    }
  }

  const handleCancelCapture = async () => {
    // Revert viewport FOV
    if (originalViewportFov !== null) {
      await updateViewportFov(originalViewportFov)
    }
    setShowIntrinsicsDropdown(false)
  }

  const frames = Array.from({ length: numFrames }, (_, i) => i + 1)

  if (!isLoaded) {
    return (
      <div className="timeline-container timeline-disabled">
        <div className="timeline-message">
          Load a reconstruction folder to enable timeline
        </div>
      </div>
    )
  }

  return (
    <div className="timeline-container">
      <div className="timeline-header">
        <h3>Camera timeline</h3>
      </div>


      <div className="timeline-content">
        {/* Left column: Timeline rows with two-column layout */}
        <div className="timeline-rows-container">
          {/* Labels column - fixed */}
          <div className="timeline-labels-column">
            <div className="row-label">
              Frame{' '}<span className="frame-counter">{currentFrame - 1}</span>
            </div>
            <div className="row-label">New</div>
            <div className="row-label">Interpolated</div>
            <div className="row-label">Original</div>
          </div>

          {/* Pills column - scrollable */}
          <div className="timeline-pills-column">
            {/* Row 0: Frame Scrubber */}
            <div 
              className="timeline-scrubber"
              onMouseDown={handleScrubberMouseDown}
            >
              {frames.map(frame => (
                <div 
                  key={frame} 
                  className={`scrubber-tick ${frame === currentFrame ? 'scrubber-tick-current' : ''}`}
                />
              ))}
              <div 
                className="scrubber-shadow"
                style={{ left: `${((currentFrame - 1) / numFrames) * 100}%`, width: `${100 / numFrames}%` }}
              />
            </div>

            {/* Row 1: New Keyframes */}
            <div className="row-pills">
              {frames.map(frame => renderPill(frame, 'new'))}
            </div>

            {/* Row 2: Interpolated */}
            <div className="row-pills">
              {frames.map(frame => renderPill(frame, 'interpolated'))}
            </div>

            {/* Row 3: Original */}
            <div className="row-pills">
              {frames.map(frame => renderPill(frame, 'original'))}
            </div>
          </div>
        </div>

        {/* Right column: Controls */}
        <div className="timeline-controls">
          {/* Two-column grid layout */}
          <div className="controls-grid">
            {/* Row 1 Col 1 - Snap + Follow */}
            <div className="control-card control-card-horizontal">
              <button className="control-btn-main" onClick={handleSnapCamera}>
                <span className="btn-text">Snap to working camera</span>
              </button>
              <label className="follow-toggle">
                <input
                  type="checkbox"
                  checked={followCamera}
                  onChange={handleFollowCamera}
                />
                <span>Auto-follow camera</span>
              </label>
            </div>

            {/* Row 1 Col 2 - Visualization toggles */}
            <div className="control-card control-card-horizontal">
              <label className="follow-toggle">
                <input
                  type="checkbox"
                  checked={showStaticBg}
                  disabled={!hasStaticBg || !isLoaded}
                  onChange={(e) => handleStaticBgToggle(e.target.checked)}
                />
                <span>Static pixel temporal persistence</span>
              </label>
              {isLoaded && !hasStaticBg && (
                <span className="input-hint">No dynamic_mask/ found</span>
              )}
              <label className="follow-toggle">
                <input
                  type="checkbox"
                  checked={showSourceCameras}
                  disabled={!isLoaded}
                  onChange={(e) => handleSourceCamerasToggle(e.target.checked)}
                />
                <span>Show source cameras</span>
              </label>
              {hasDse && (
                <label className="follow-toggle">
                  <input
                    type="checkbox"
                    checked={showDseCameras}
                    disabled={!isLoaded}
                    onChange={(e) => handleDseCamerasToggle(e.target.checked)}
                  />
                  <span>Show DSE cameras</span>
                </label>
              )}
            </div>

            {/* Row 2 Col 1 - Capture + Factory Reset buttons */}
            <div className="control-card control-card-buttons">
              <div className="capture-container">
                <button 
                  className="control-btn-main primary" 
                  onClick={handleToggleCaptureDropdown}
                  disabled={!frame1Intrinsics}
                  title="Capture current view (click to adjust FOV)"
                >
                  <span className="btn-text">Capture current view</span>
                </button>

                {showIntrinsicsDropdown && frame1Intrinsics && (
                  <div className="intrinsics-dropdown">
                    <div className="dropdown-label">Adjust FOV (live preview)</div>
                    <div className="dropdown-fov-control">
                      <span className="fov-multiplier">{fovMultiplier.toFixed(2)}x</span>
                      <input
                        type="range"
                        min="0.1"
                        max="5.0"
                        step="0.01"
                        value={fovMultiplier}
                        onChange={(e) => handleFovMultiplierChange(parseFloat(e.target.value))}
                        className="dropdown-fov-slider"
                      />
                    </div>
                    <div className="dropdown-actions">
                      <button 
                        className="dropdown-btn dropdown-btn-cancel"
                        onClick={handleCancelCapture}
                      >
                        Cancel
                      </button>
                      <button 
                        className="dropdown-btn dropdown-btn-confirm"
                        onClick={handleConfirmCapture}
                      >
                        Confirm
                      </button>
                    </div>
                  </div>
                )}
              </div>

              <div className="reset-buttons-container">
                <button className="control-btn-main danger reset-btn-half" onClick={handleFactoryReset}>
                  <span className="btn-text">Factory reset</span>
                </button>
                <button className="control-btn-main danger reset-btn-half" onClick={handleResetKeyframes}>
                  <span className="btn-text">Reset keyframes</span>
                </button>
              </div>
            </div>

            {/* Row 2 Col 2 - Point Size Control */}
            <div className="control-card control-card-confidence">
              <div className="confidence-header-compact">
                <label className="confidence-label-compact">
                  <span>Point size</span>
                  <span className="confidence-value">{(localPointSize).toFixed(4)}</span>
                </label>
                <button 
                  className={`control-btn-tiny ${pointSizeDirty ? 'primary' : ''}`}
                  onClick={handleUpdatePointSize}
                  disabled={!pointSizeDirty || updatingPointSize}
                  title="Update point cloud size"
                >
                  {updatingPointSize ? '...' : 'U'}
                </button>
              </div>
              <input
                type="range"
                min="0"
                max="0.01"
                step="0.0001"
                value={localPointSize}
                onChange={(e) => handlePointSizeChange(parseFloat(e.target.value))}
                className={`confidence-slider-compact ${pointSizeDirty ? 'confidence-slider-dirty' : ''}`}
              />
            </div>
          </div>

          {/* Playback and Tension Islands - Side by Side */}
          <div className="islands-row">
            {/* Playback Controls */}
            <div className="playback-control-island">
              <button 
                className="control-btn-play"
                onClick={handlePlayPause}
                title={isPlaying ? "Pause" : "Play"}
              >
                {isPlaying ? '||' : '>'}
              </button>
              <input
                type="number"
                min="0"
                max={numFrames - 1}
                value={currentFrame - 1}
                onChange={(e) => {
                  const z = Math.max(0, Math.min(numFrames - 1, parseInt(e.target.value) || 0))
                  onFrameClick(z + 1)
                }}
                className="frame-input"
                title="Current Frame (0-indexed)"
              />
              <input
                type="number"
                min="1"
                max="60"
                value={playbackFps}
                onChange={(e) => handleFpsChange(Math.max(1, Math.min(60, parseInt(e.target.value) || 30)))}
                className="fps-input"
                title="Playback FPS"
              />
              <span className="fps-label">fps</span>
            </div>

            {/* Tension Control Island */}
            <div className="tension-control-island">
              <div className="tension-header-compact">
                <label className="tension-label-compact">
                  <span>Catmull-Rom tension</span>
                  <span className="tension-value">{localTension.toFixed(2)}</span>
                </label>
                <button
                  className={`control-btn-tiny ${tensionDirty ? 'primary' : ''}`}
                  onClick={handleRecomputeTension}
                  disabled={!tensionDirty || recomputing}
                  title="Recompute trajectory with new tension"
                >
                  {recomputing ? '...' : 'R'}
                </button>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={localTension}
                onChange={(e) => handleTensionChange(parseFloat(e.target.value))}
                className={`tension-slider-compact ${tensionDirty ? 'tension-slider-dirty' : ''}`}
              />
            </div>

            {/* DSE Frame Interval (only when DSE loaded) */}
            {hasDse && (
              <div className="tension-control-island">
                <div className="tension-header-compact">
                  <label className="tension-label-compact">
                    <span>DSE frame interval</span>
                    <span className="tension-value">{localDseFrameInterval}</span>
                  </label>
                  <button
                    className={`control-btn-tiny ${dseFrameIntervalDirty ? 'primary' : ''}`}
                    onClick={handleUpdateDseFrameInterval}
                    disabled={!dseFrameIntervalDirty || updatingDseFrameInterval}
                    title="Rebuild static background at new DSE stride"
                  >
                    {updatingDseFrameInterval ? '...' : 'U'}
                  </button>
                </div>
                <input
                  type="range"
                  min="1"
                  max={Math.max(1, numDseFrames)}
                  step="1"
                  value={localDseFrameInterval}
                  onChange={(e) => handleDseFrameIntervalChange(parseInt(e.target.value))}
                  className={`tension-slider-compact ${dseFrameIntervalDirty ? 'tension-slider-dirty' : ''}`}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

/**
 * Compute interpolation segments from new keyframes
 * @param newKeyframes - Map of frame number to keyframe data
 * @param numFrames - Total number of frames in the NPZ file
 * @returns Array of segments {a, b, kind}
 */
function computeSegments(newKeyframes: Keyframes, numFrames: number): Segment[] {
  const keyframeFrames = Object.keys(newKeyframes).map(Number).sort((a, b) => a - b)
  
  if (keyframeFrames.length === 0) {
    // No keyframes: no interpolation
    return []
  }

  // Build breakpoints: [1, ...keyframes..., numFrames]
  const breakpoints: Breakpoint[] = []
  
  const maxFrame = numFrames  // Use actual NPZ frame count
  
  // Add frame 1 (start)
  if (!keyframeFrames.includes(1)) {
    breakpoints.push({ frame: 1, isNew: false })
  } else {
    breakpoints.push({ frame: 1, isNew: true })
  }
  
  // Add keyframes (excluding 1 and maxFrame if already added)
  for (const frame of keyframeFrames) {
    if (frame !== 1 && frame !== maxFrame) {
      breakpoints.push({ frame, isNew: true })
    }
  }
  
  // Add final frame (maxFrame)
  if (!keyframeFrames.includes(maxFrame)) {
    breakpoints.push({ frame: maxFrame, isNew: false })
  } else {
    breakpoints.push({ frame: maxFrame, isNew: true })
  }

  // Sort breakpoints
  breakpoints.sort((a, b) => a.frame - b.frame)

  // Create segments
  const segments: Segment[] = []
  for (let i = 0; i < breakpoints.length - 1; i++) {
    const a = breakpoints[i]
    const b = breakpoints[i + 1]
    
    // Only create segment if there are interior frames
    if (b.frame - a.frame <= 1) continue

    // Determine segment type based on start and end
    let kind: 'OldNew' | 'NewOld' | 'NewNew'
    if (a.isNew && b.isNew) {
      kind = 'NewNew'
    } else if (a.isNew && !b.isNew) {
      kind = 'NewOld'
    } else {
      kind = 'OldNew'
    }
    
    segments.push({
      a: a.frame,
      b: b.frame,
      kind
    })
  }

  return segments
}

export default Timeline
export type { TimelineProps, Keyframes, CameraPose }

