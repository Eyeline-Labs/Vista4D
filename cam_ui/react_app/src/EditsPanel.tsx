import React, { useEffect, useRef, useState } from 'react'
import './EditsPanel.css'

export type TargetKind = 'existing' | 'insert' | 'duplicate'
export type OpKind = 'translate' | 'rotate' | 'scale' | 'remove'
export type Scope = 'global' | 'frame'

export interface EditOp {
  op: OpKind
  // translate/rotate: [x,y,z]; scale: number | [x,y,z]; remove: no params
  params?: number[] | number
}

export interface EditTarget {
  kind: TargetKind
  prompt: string
  source?: string  // Required when kind === 'insert'
}

export interface Edit {
  target: EditTarget
  ops: EditOp[]
  scope: Scope
  mask_expansion?: [number, number]
  centroid_threshold?: number
}

interface Props {
  isLoaded: boolean
  editsReady: boolean
}

// Hard ceiling on a single apply. Real-world inserts do load insert recon_and_seg + preprocess +
// load SAM3 (~30s cold) + run SAM3 on both scenes, which can stack to several minutes on a GPU.
// 15 min leaves headroom for heavy scenes while still catching a truly hung CPU-only SAM3 run.
const APPLY_TIMEOUT_MS = 900_000

const DEFAULT_PARAMS: Record<OpKind, number[] | number | undefined> = {
  translate: [0, 0, 0],
  rotate: [0, 0, 0],
  scale: 1,
  remove: undefined,
}

// An edit is "runnable" only once the user has filled in the target. Incomplete edits are
// hidden from the backend PUT so that an in-progress draft doesn't trigger SAM3 loading /
// insert-source preprocessing on an empty prompt.
function isRunnable(edit: Edit): boolean {
  const prompt = edit.target.prompt.trim()
  if (!prompt) return false
  if (edit.target.kind === 'insert' && !(edit.target.source ?? '').trim()) return false
  return true
}

function newEdit(): Edit {
  return {
    target: { kind: 'existing', prompt: '' },
    ops: [{ op: 'translate', params: [0, 0, 0] }],
    scope: 'global',
  }
}

// Number input with local string state.
//
// Why local string state: native `<input type="number" value={n}>` round-trips parent state
// through `String(n)` on every keystroke. That clobbers in-progress text — typing "1" into a
// "0" field appends to "01" because the underlying value is still 0 mid-keystroke. A local
// text buffer lets the user type freely; we parse and propagate to the parent on each change
// so the draft stays current, and we re-sync the local text when the parent value changes
// from outside (e.g., on Apply, scene reload).
//
// `onFocus={selectAll}` matches the user's mental model: clicking a "0" field and typing
// replaces the 0 instead of inserting next to it.
function NumberInput({
  value, onChange, step, className, min, max,
}: {
  value: number
  onChange: (v: number) => void
  step?: number
  className?: string
  min?: number
  max?: number
}) {
  const [text, setText] = useState(String(value))
  const lastValueRef = useRef(value)
  useEffect(() => {
    if (value !== lastValueRef.current) {
      setText(String(value))
      lastValueRef.current = value
    }
  }, [value])

  return (
    <input
      type="number"
      step={step}
      min={min}
      max={max}
      className={className}
      value={text}
      onChange={(e) => {
        const next = e.target.value
        setText(next)
        // Only push a parsed number to the parent when the buffer is currently a valid number.
        // Intermediate states like "" or "-" or "1." stay local until they parse cleanly.
        const parsed = parseFloat(next)
        if (!isNaN(parsed)) {
          lastValueRef.current = parsed
          onChange(parsed)
        }
      }}
      onBlur={() => {
        // Snap the text to canonical form on leave. If parse failed entirely, restore last value.
        const parsed = parseFloat(text)
        if (isNaN(parsed)) setText(String(value))
        else setText(String(parsed))
      }}
      onFocus={(e) => e.target.select()}
    />
  )
}

function numberArrayInput(
  values: number[],
  onChange: (v: number[]) => void,
  labels: string[],
) {
  return (
    <div className="edit-vec">
      {values.map((v, i) => (
        <label key={i} className="edit-vec-item">
          <span>{labels[i]}</span>
          <NumberInput
            step={0.01}
            value={v}
            onChange={(parsed) => {
              const next = [...values]
              next[i] = parsed
              onChange(next)
            }}
          />
        </label>
      ))}
    </div>
  )
}

function OpEditor({
  op, onChange, onRemove,
}: {
  op: EditOp
  onChange: (op: EditOp) => void
  onRemove: () => void
}) {
  const handleKindChange = (kind: OpKind) => {
    onChange({ op: kind, params: DEFAULT_PARAMS[kind] })
  }

  return (
    <div className="edit-op">
      <div className="edit-op-header">
        <select
          value={op.op}
          onChange={(e) => handleKindChange(e.target.value as OpKind)}
          className="edit-select"
        >
          <option value="translate">translate</option>
          <option value="rotate">rotate</option>
          <option value="scale">scale</option>
          <option value="remove">remove</option>
        </select>
        <button className="edit-icon-btn" onClick={onRemove} title="Delete operation">×</button>
      </div>
      {op.op === 'translate' && Array.isArray(op.params) && numberArrayInput(
        op.params, (v) => onChange({ ...op, params: v }), ['x', 'y', 'z'],
      )}
      {op.op === 'rotate' && Array.isArray(op.params) && numberArrayInput(
        op.params, (v) => onChange({ ...op, params: v }), ['rx°', 'ry°', 'rz°'],
      )}
      {op.op === 'scale' && (
        <div className="edit-scale-row">
          <label className="edit-scale-mode">
            <input
              type="checkbox"
              checked={Array.isArray(op.params)}
              onChange={(e) => {
                if (e.target.checked) {
                  const s = typeof op.params === 'number' ? op.params : 1
                  onChange({ ...op, params: [s, s, s] })
                } else {
                  const s = Array.isArray(op.params) ? op.params[0] : 1
                  onChange({ ...op, params: s })
                }
              }}
            />
            <span>per-axis</span>
          </label>
          {Array.isArray(op.params)
            ? numberArrayInput(op.params, (v) => onChange({ ...op, params: v }), ['sx', 'sy', 'sz'])
            : (
              <NumberInput
                step={0.01}
                value={typeof op.params === 'number' ? op.params : 1}
                className="edit-scalar"
                onChange={(parsed) => onChange({ ...op, params: parsed })}
              />
            )}
        </div>
      )}
      {op.op === 'remove' && (
        <div className="edit-op-note">Drops the selected points. No parameters.</div>
      )}
    </div>
  )
}

function EditCard({
  edit, index, total, onChange, onDelete, onMove,
}: {
  edit: Edit
  index: number
  total: number
  onChange: (e: Edit) => void
  onDelete: () => void
  onMove: (delta: number) => void
}) {
  const [showAdvanced, setShowAdvanced] = useState(false)

  const updateOp = (i: number, next: EditOp) => {
    const ops = [...edit.ops]
    ops[i] = next
    onChange({ ...edit, ops })
  }

  const removeOp = (i: number) => {
    const ops = edit.ops.filter((_, j) => j !== i)
    onChange({ ...edit, ops })
  }

  const addOp = () => {
    onChange({ ...edit, ops: [...edit.ops, { op: 'translate', params: [0, 0, 0] }] })
  }

  const updateTarget = (next: Partial<EditTarget>) => {
    onChange({ ...edit, target: { ...edit.target, ...next } })
  }

  return (
    <div className="edit-card">
      <div className="edit-card-head">
        <span className="edit-card-index">#{index + 1}</span>
        <div className="edit-card-head-actions">
          <button
            className="edit-icon-btn"
            onClick={() => onMove(-1)}
            disabled={index === 0}
            title="Move up"
          >↑</button>
          <button
            className="edit-icon-btn"
            onClick={() => onMove(1)}
            disabled={index === total - 1}
            title="Move down"
          >↓</button>
          <button className="edit-icon-btn edit-icon-danger" onClick={onDelete} title="Delete edit">×</button>
        </div>
      </div>

      <div className="edit-field">
        <label>Target kind</label>
        <select
          value={edit.target.kind}
          onChange={(e) => updateTarget({ kind: e.target.value as TargetKind })}
          className="edit-select"
        >
          <option value="existing">existing</option>
          <option value="duplicate">duplicate</option>
          <option value="insert">insert</option>
        </select>
      </div>

      {/* Insert source first when kind=insert: the prompt segments whatever source was loaded. */}
      {edit.target.kind === 'insert' && (
        <div className="edit-field">
          <label>
            Insert source <span className="edit-hint">(path to recon_and_seg folder)</span>
          </label>
          <input
            type="text"
            value={edit.target.source ?? ''}
            onChange={(e) => updateTarget({ source: e.target.value })}
            placeholder="results/single/other-scene/recon_and_seg"
          />
        </div>
      )}

      <div className="edit-field">
        <label>
          Prompt <span className="edit-hint">(comma-separated keywords for SAM3)</span>
        </label>
        <input
          type="text"
          value={edit.target.prompt}
          onChange={(e) => updateTarget({ prompt: e.target.value })}
          placeholder="person, dog"
        />
      </div>

      <div className="edit-field">
        <label>Scope</label>
        <select
          value={edit.scope}
          onChange={(e) => onChange({ ...edit, scope: e.target.value as Scope })}
          className="edit-select"
        >
          <option value="global">global</option>
          <option value="frame">frame</option>
        </select>
      </div>

      <div className="edit-field">
        <label>Operations</label>
        <div className="edit-ops-list">
          {edit.ops.map((op, i) => (
            <OpEditor
              key={i}
              op={op}
              onChange={(next) => updateOp(i, next)}
              onRemove={() => removeOp(i)}
            />
          ))}
        </div>
        <button className="edit-small-btn" onClick={addOp}>+ Add operation</button>
      </div>

      <button
        className="edit-advanced-toggle"
        onClick={() => setShowAdvanced(!showAdvanced)}
      >
        {showAdvanced ? '▾' : '▸'} Advanced
      </button>
      {showAdvanced && (
        <div className="edit-advanced">
          <div className="edit-field">
            <label>Mask expansion <span className="edit-hint">[radius, iterations]</span></label>
            {numberArrayInput(
              edit.mask_expansion ?? [0, 0],
              (v) => onChange({ ...edit, mask_expansion: [v[0], v[1]] }),
              ['r', 'i'],
            )}
          </div>
          <div className="edit-field">
            <label>Centroid threshold <span className="edit-hint">(quantile, default 0.6)</span></label>
            <NumberInput
              step={0.05}
              min={0}
              max={1}
              value={edit.centroid_threshold ?? 0.6}
              className="edit-scalar"
              onChange={(parsed) => onChange({ ...edit, centroid_threshold: parsed })}
            />
          </div>
        </div>
      )}
    </div>
  )
}

export default function EditsPanel({ isLoaded, editsReady }: Props) {
  const [open, setOpen] = useState(false)
  // Two-state split. `committedEdits` is what's currently applied on the backend (the source
  // of truth that mirrors what `render_edit.py` would produce for export). `draftEdits` is
  // what the panel UI is editing locally — typing in number fields, swapping target kinds,
  // adding/removing ops all stay in the draft and only cross to the backend when the user
  // clicks Apply. Until Apply, no PUT fires, so heavy work (SAM3, unproject_insert,
  // per-frame point cloud rebuilds) never runs on partial input.
  const [committedEdits, setCommittedEdits] = useState<Edit[]>([])
  const [draftEdits, setDraftEdits] = useState<Edit[]>([])
  const [status, setStatus] = useState<'idle' | 'applying' | 'error'>('idle')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [progressMessage, setProgressMessage] = useState<string | null>(null)
  const [exportFilename, setExportFilename] = useState('edits.json')
  const [exportMessage, setExportMessage] = useState<string | null>(null)

  // Poll /api/edits/progress while an apply is in flight so the user sees *what* is happening
  // (SAM3 model load, segmentation, insert scene load, ...) instead of a frozen "Applying...".
  useEffect(() => {
    if (status !== 'applying') {
      setProgressMessage(null)
      return
    }
    let cancelled = false
    const tick = async () => {
      try {
        const res = await fetch('/api/edits/progress')
        if (!res.ok) return
        const data = await res.json()
        if (cancelled) return
        const msg = typeof data.progress === 'string' ? data.progress : null
        setProgressMessage(msg && msg !== 'idle' ? msg : null)
      } catch { /* polling; ignore transient errors */ }
    }
    tick()
    const id = setInterval(tick, 500)
    return () => { cancelled = true; clearInterval(id) }
  }, [status])

  // Fetch current edits on mount and whenever the scene changes. Both committed and draft
  // start in sync — the user begins with a clean slate that matches the backend.
  useEffect(() => {
    if (!isLoaded) {
      setCommittedEdits([])
      setDraftEdits([])
      return
    }
    fetch('/api/edits')
      .then((r) => r.json())
      .then((data) => {
        const loaded = Array.isArray(data.edits) ? data.edits : []
        setCommittedEdits(loaded)
        setDraftEdits(loaded)
      })
      .catch((err) => console.error('Failed to fetch edits:', err))
  }, [isLoaded])

  const handleApply = async () => {
    if (status === 'applying') return
    const runnable = draftEdits.filter(isRunnable)
    setStatus('applying')
    setErrorMessage(null)
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), APPLY_TIMEOUT_MS)
    try {
      const res = await fetch('/api/edits', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ edits: runnable }),
        signal: controller.signal,
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
        throw new Error(err.detail || `HTTP ${res.status}`)
      }
      // Commit the exact draft we just sent — pending state cleared on success.
      setCommittedEdits(draftEdits)
      setStatus('idle')
    } catch (err) {
      setStatus('error')
      const isAbort = err instanceof DOMException && err.name === 'AbortError'
      if (isAbort) {
        setErrorMessage(
          `Apply timed out after ${APPLY_TIMEOUT_MS / 1000}s — check the backend console ` +
          `(SAM3 needs a GPU; CPU-only runs will hang).`
        )
      } else {
        setErrorMessage(err instanceof Error ? err.message : String(err))
      }
    } finally {
      clearTimeout(timeoutId)
    }
  }

  const addEdit = () => setDraftEdits([...draftEdits, newEdit()])
  const updateEdit = (i: number, next: Edit) => {
    const copy = [...draftEdits]
    copy[i] = next
    setDraftEdits(copy)
  }
  const deleteEdit = (i: number) => setDraftEdits(draftEdits.filter((_, j) => j !== i))
  const moveEdit = (i: number, delta: number) => {
    const j = i + delta
    if (j < 0 || j >= draftEdits.length) return
    const copy = [...draftEdits]
    const [moved] = copy.splice(i, 1)
    copy.splice(j, 0, moved)
    setDraftEdits(copy)
  }

  const handleExport = async () => {
    setExportMessage(null)
    try {
      const res = await fetch('/api/edits/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: exportFilename }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setExportMessage(`Saved ${data.num_edits} edit(s) → ${data.path}`)
    } catch (err) {
      setExportMessage(`Export failed: ${err instanceof Error ? err.message : String(err)}`)
    }
  }

  // Pending = the draft differs from what's currently committed on the backend. Cheap
  // structural compare via JSON.stringify is fine for the small edit lists we deal with.
  const isPending = JSON.stringify(draftEdits) !== JSON.stringify(committedEdits)

  let statusLabel = 'Ready'
  let statusClass = 'edits-status-ready'
  if (!isLoaded) {
    statusLabel = 'No scene'
    statusClass = 'edits-status-off'
  } else if (!editsReady) {
    statusLabel = 'Building buffers...'
    statusClass = 'edits-status-loading'
  } else if (status === 'applying') {
    statusLabel = 'Applying...'
    statusClass = 'edits-status-loading'
  } else if (status === 'error') {
    statusLabel = 'Error'
    statusClass = 'edits-status-error'
  } else if (isPending) {
    statusLabel = 'Pending'
    statusClass = 'edits-status-loading'
  }

  return (
    <aside className={`edits-panel ${open ? 'edits-panel-open' : 'edits-panel-closed'}`}>
      <button
        className="edits-panel-toggle"
        onClick={() => setOpen(!open)}
        title={
          status === 'error'
            ? `Edits error: ${errorMessage ?? 'see panel'}`
            : (open ? 'Close edits panel' : 'Open edits panel')
        }
      >
        {open ? '▸' : '◂'}
        <span className="edits-panel-toggle-label">Edits</span>
        {!open && status === 'error' && <span className="edits-panel-toggle-dot edits-panel-toggle-dot-error" />}
        {!open && status === 'applying' && <span className="edits-panel-toggle-dot edits-panel-toggle-dot-busy" />}
        {!open && isPending && status !== 'error' && status !== 'applying' && (
          <span className="edits-panel-toggle-dot edits-panel-toggle-dot-busy" />
        )}
      </button>
      {open && (
        <div className="edits-panel-body">
          <div className="edits-panel-head">
            <h2>Edits</h2>
            <span className={`edits-status ${statusClass}`}>{statusLabel}</span>
          </div>
          {status === 'applying' && (
            <div className="edits-progress">
              {progressMessage ?? 'Applying edits...'}
            </div>
          )}
          {errorMessage && (
            <div className="edits-error">{errorMessage}</div>
          )}
          <div className="edits-list">
            {draftEdits.map((edit, i) => (
              <EditCard
                key={i}
                edit={edit}
                index={i}
                total={draftEdits.length}
                onChange={(next) => updateEdit(i, next)}
                onDelete={() => deleteEdit(i)}
                onMove={(delta) => moveEdit(i, delta)}
              />
            ))}
            {draftEdits.length === 0 && (
              <div className="edits-empty">No edits yet. Click "+ Add edit" to compose the scene.</div>
            )}
          </div>
          <div className="edits-actions">
            <button
              className="edits-add-btn"
              onClick={addEdit}
              disabled={!isLoaded}
            >+ Add edit</button>
            <button
              className="edits-apply-btn"
              onClick={handleApply}
              disabled={!isLoaded || !editsReady || status === 'applying' || !isPending}
              title={
                !isPending
                  ? 'No pending changes'
                  : status === 'applying'
                    ? 'Applying...'
                    : 'Apply pending edits to the scene (runs SAM3 / unprojects inserts as needed)'
              }
            >
              {status === 'applying' ? 'Applying…' : 'Apply edits'}
            </button>
          </div>

          <div className="edits-export">
            <label>Export filename</label>
            <input
              type="text"
              value={exportFilename}
              onChange={(e) => setExportFilename(e.target.value)}
              placeholder="edits.json"
            />
            <button
              className="edits-export-btn"
              onClick={handleExport}
              disabled={!isLoaded || committedEdits.length === 0 || isPending}
              title={
                isPending
                  ? 'Apply pending changes before exporting'
                  : (committedEdits.length === 0 ? 'No applied edits to export' : 'Export the currently applied edits to JSON')
              }
            >Export edits</button>
            {exportMessage && <div className="edits-export-msg">{exportMessage}</div>}
          </div>
        </div>
      )}
    </aside>
  )
}
