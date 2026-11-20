import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import * as api from "../api";
import { JobProgress } from "../components/JobProgress";
import { ProcessingModeSelect, resolveProcessingMode } from "../components/ProcessingModeSelect";
import { RegionPicker } from "../components/RegionPicker";
import { StemMixer } from "../components/StemMixer";
import { practiceFromJob } from "./isolatePractice";
import type { StemInfo } from "../mixer/engine";
import {
  fullFileTrimNotice,
  isolateRepairBody,
  isolateStartDisabled,
  isolateStartLabel,
} from "./isolateSubmit";
import { StableLabel } from "../pending";
import {
  DEFAULT_TRACK_OPTIONS,
  DEMUCS_STEM_CHECKBOX_IDS,
  GUITAR_TRACK_OPTION_IDS,
  HOSTED_FILE_ONLY_NOTE,
  TRACK_OPTIONS,
  VOCALS_INSTRUMENTAL_OPTION_ID,
  type TrackOptionId,
} from "../trackOptions";

const MIN_REGION_SEC = 5;

const STEM_LABELS: Record<string, string> = {
  vocals: "Vocals",
  drums: "Drums",
  bass: "Bass",
  other: "Other",
  guitar: "Guitar",
  piano: "Piano",
  lead_guitar: "Lead (legacy)",
  rhythm_guitar: "Rhythm (legacy)",
  guitar1: "Guitar 1",
  guitar2: "Guitar 2",
  metronome: "Metronome",
};

const ROFORMER_DOWNLOAD_CAVEAT =
  "Downloads ~700 MB BS-RoFormer-SW weights on first use, then a guitar specialist (~45 MB). Much slower on CPU. Residual bleed remains. Community weights have no stated license. Use accordingly.";
const ROFORMER_MIXED_STEMS_NOTE =
  "All stems are separated in one BS-RoFormer pass; unselected stems are discarded.";
const TRACKS_PICKER_HELP =
  "Pick the stems you want. Guitar can use Demucs (faster) or BS-RoFormer (cleaner on a GPU, much slower on CPU). Neither is bleed-free, and piano from Demucs is unreliable.";

type ResolvedTrackSelection = {
  model: string;
  two_stems: string | null;
  emit_stems: string[];
  fold_other_into_guitar: boolean;
  guitar_refine: boolean;
  caveat: string;
};

function trackSelectionCaveat(optionIds: string[], usesRoformer: boolean): string {
  const parts: string[] = [];
  if (usesRoformer) {
    parts.push(ROFORMER_DOWNLOAD_CAVEAT);
    if (optionIds.some((id) => !GUITAR_TRACK_OPTION_IDS.has(id as TrackOptionId))) {
      parts.push(ROFORMER_MIXED_STEMS_NOTE);
    }
  }
  return parts.join(" ");
}

function resolveTrackSelection(optionIds: string[]): ResolvedTrackSelection {
  const ids = optionIds.filter((id): id is TrackOptionId => id in TRACK_OPTIONS);
  if (ids.length === 0) {
    throw new Error("Pick at least one track to separate.");
  }

  const guitarPicks = ids.filter((id) => GUITAR_TRACK_OPTION_IDS.has(id));
  if (guitarPicks.length > 1) {
    throw new Error("Pick only one guitar option.");
  }

  if (ids.includes(VOCALS_INSTRUMENTAL_OPTION_ID)) {
    if (ids.length > 1) {
      throw new Error("Vocals & instrumental cannot combine with other tracks.");
    }
    return {
      model: "htdemucs",
      two_stems: "vocals",
      emit_stems: ["vocals"],
      fold_other_into_guitar: true,
      guitar_refine: false,
      caveat: "",
    };
  }

  const emit: string[] = [];
  const customStems: string[] = [];
  for (const oid of ids) {
    const stem = TRACK_OPTIONS[oid].stem;
    if (!emit.includes(stem)) emit.push(stem);
    if (!customStems.includes(stem)) customStems.push(stem);
  }

  const usesRoformer =
    guitarPicks.length === 1 &&
    (guitarPicks[0] === "guitar_roformer" || guitarPicks[0] === "guitar_roformer_refine");
  const guitarRefine = guitarPicks[0] === "guitar_roformer_refine";

  let model: string;
  if (usesRoformer) {
    model = "bs_roformer_sw";
  } else if (customStems.includes("guitar") || customStems.includes("piano")) {
    model = "htdemucs_6s";
  } else {
    model = "htdemucs";
  }

  return {
    model,
    two_stems: null,
    emit_stems: emit,
    fold_other_into_guitar: !customStems.includes("other"),
    guitar_refine: guitarRefine,
    caveat: trackSelectionCaveat(ids, usesRoformer),
  };
}

const AUDIO_KINDS = new Set(Object.keys(STEM_LABELS));

function formatTime(sec: number): string {
  if (!isFinite(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function formatRegionLabel(startSec: number, endSec: number): string {
  return `${formatTime(startSec)}–${formatTime(endSec)}`;
}

function isDiagnosticKind(kind: string): boolean {
  return kind.endsWith("_diagnostics");
}

type GuitarTrackChoice = "none" | TrackOptionId;

export function IsolatePage() {
  const [file, setFile] = useState<File | null>(null);
  const [demucsChecked, setDemucsChecked] = useState<Record<TrackOptionId, boolean>>({
    vocals_demucs: true,
    drums_demucs: false,
    bass_demucs: false,
    piano_demucs: false,
    other_demucs: false,
    guitar_demucs_6s: false,
    guitar_roformer: false,
    guitar_roformer_refine: false,
    vocals_instrumental_demucs: false,
  });
  const [guitarTrack, setGuitarTrack] = useState<GuitarTrackChoice>("guitar_demucs_6s");
  const [vocalsInstrumentalOnly, setVocalsInstrumentalOnly] = useState(false);
  const [processingMode, setProcessingMode] = useState("auto");
  const [lowEndRestoreDb, setLowEndRestoreDb] = useState(0);
  const [subBassDebleed, setSubBassDebleed] = useState(false);
  const [capabilities, setCapabilities] = useState<api.SystemCapabilities | null>(null);
  const [duration, setDuration] = useState<number | null>(null);
  const [useRegion, setUseRegion] = useState(false);
  const [regionStart, setRegionStart] = useState(0);
  const [regionEnd, setRegionEnd] = useState(30);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<api.JobResponse | null>(null);
  const [jobList, setJobList] = useState<api.JobResponse[]>([]);
  const [stopWatch, setStopWatch] = useState<(() => void) | null>(null);
  const [clipLabel, setClipLabel] = useState<string | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const previewStopRef = useRef<(() => void) | null>(null);
  const [objectUrl, setObjectUrl] = useState<string | null>(null);

  const trackOptions = useMemo((): TrackOptionId[] => {
    if (vocalsInstrumentalOnly) {
      return [VOCALS_INSTRUMENTAL_OPTION_ID];
    }
    const options = DEMUCS_STEM_CHECKBOX_IDS.filter((id) => demucsChecked[id]);
    if (guitarTrack !== "none") {
      options.push(guitarTrack);
    }
    return options;
  }, [demucsChecked, guitarTrack, vocalsInstrumentalOnly]);

  const resolved = useMemo((): ResolvedTrackSelection => {
    try {
      return resolveTrackSelection(trackOptions);
    } catch {
      return resolveTrackSelection(DEFAULT_TRACK_OPTIONS);
    }
  }, [trackOptions]);

  const pickerError = useMemo(() => {
    try {
      resolveTrackSelection(trackOptions);
      return null;
    } catch (err) {
      return err instanceof Error ? err.message : String(err);
    }
  }, [trackOptions]);

  useEffect(() => {
    let cancelled = false;
    api
      .getSystemCapabilities()
      .then((c) => {
        if (!cancelled) setCapabilities(c);
      })
      .catch(() => {
        /* fallback modes used by ProcessingModeSelect */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const refresh = () => {
      api
        .listJobs({ kind: "isolate", limit: 10 })
        .then((rows) => {
          if (!cancelled) setJobList(rows);
        })
        .catch(() => {
          /* ignore */
        });
    };
    refresh();
    const t = window.setInterval(refresh, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(t);
    };
  }, [job?.id, job?.status]);

  useEffect(() => {
    previewStopRef.current?.();
    previewStopRef.current = null;
    setDuration(null);
    setRegionStart(0);
    setRegionEnd(30);
    if (!file) {
      setObjectUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setObjectUrl(url);
    return () => {
      previewStopRef.current?.();
      URL.revokeObjectURL(url);
    };
  }, [file]);

  const selectedMode = useMemo(() => {
    return resolveProcessingMode(capabilities, processingMode);
  }, [capabilities, processingMode]);

  const regionLength = useRegion ? Math.max(0, regionEnd - regionStart) : null;
  const modeCap = selectedMode?.max_duration_sec ?? null;
  const regionOverCap =
    useRegion && regionLength !== null && modeCap !== null && regionLength > modeCap;
  const trimNotice = fullFileTrimNotice(duration, modeCap, useRegion);

  const regionLabel =
    useRegion && duration !== null ? formatRegionLabel(regionStart, regionEnd) : null;

  useEffect(() => {
    if (duration === null && useRegion) setUseRegion(false);
  }, [duration, useRegion]);

  function onAudioMeta() {
    const audio = audioRef.current;
    if (!audio) return;
    const d = audio.duration;
    if (!isFinite(d) || d <= 0) {
      setDuration(null);
      return;
    }
    setDuration(d);
    const end = Math.min(d, Math.max(MIN_REGION_SEC, 30));
    setRegionStart(0);
    setRegionEnd(end);
  }

  function shortenEndToCap() {
    if (modeCap === null) return;
    setRegionEnd(Math.min(duration ?? regionEnd, regionStart + modeCap));
  }

  function onRegionChange(start: number, end: number) {
    if (duration === null) return;
    let nextStart = Math.max(0, Math.min(start, duration));
    let nextEnd = Math.max(nextStart + MIN_REGION_SEC, Math.min(end, duration));
    if (nextEnd > duration) {
      nextEnd = duration;
      nextStart = Math.max(0, nextEnd - MIN_REGION_SEC);
    }
    setRegionStart(nextStart);
    setRegionEnd(nextEnd);
  }

  function toggleDemucsStem(id: TrackOptionId, checked: boolean) {
    setDemucsChecked((prev) => ({ ...prev, [id]: checked }));
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!file) {
      setError("Choose an audio file");
      return;
    }
    if (pickerError) {
      setError(pickerError);
      return;
    }
    if (useRegion) {
      if (duration === null) {
        setError("Could not read audio length. Turn off section mode or try another file");
        return;
      }
      if (regionStart >= regionEnd) {
        setError("Start must be before end");
        return;
      }
      if (regionEnd - regionStart < MIN_REGION_SEC) {
        setError(`Section must be at least ${MIN_REGION_SEC} seconds`);
        return;
      }
      if (regionEnd > duration + 0.05) {
        setError("End time exceeds file length");
        return;
      }
      if (regionOverCap) {
        setError(
          selectedMode
            ? `${selectedMode.label} allows up to ${modeCap} s per job. Shorten the section or choose a different mode.`
            : "Section is longer than this processing mode allows"
        );
        return;
      }
    }

    setBusy(true);
    setError(null);
    stopWatch?.();
    setClipLabel(
      useRegion && regionLabel
        ? regionLabel
        : trimNotice && modeCap !== null
          ? formatRegionLabel(0, modeCap)
          : null,
    );

    const body: Record<string, unknown> = {
      upload_id: "",
      model: resolved.model,
      processing_mode: processingMode,
      emit_stems: [...resolved.emit_stems],
      fold_other_into_guitar: resolved.fold_other_into_guitar,
      ...isolateRepairBody({ lowEndRestoreDb, subBassDebleed }),
    };
    if (resolved.two_stems) {
      body.two_stems = resolved.two_stems;
    }
    if (resolved.guitar_refine) {
      body.guitar_refine = true;
    }
    if (useRegion) {
      body.start_sec = regionStart;
      body.end_sec = regionEnd;
    } else if (trimNotice && modeCap !== null) {
      body.start_sec = 0;
      body.end_sec = modeCap;
    }

    try {
      const up = await api.uploadAudio(file);
      body.upload_id = up.upload_id;
      const created = await api.createIsolateJob(body);
      setJob(created);
      const stop = api.watchJob(created.id, setJob);
      setStopWatch(() => stop);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setClipLabel(null);
    } finally {
      setBusy(false);
    }
  }

  const practice = practiceFromJob(job);
  const stems: StemInfo[] = useMemo(() => {
    if (!job || job.status !== "succeeded") return [];
    return Object.entries(job.artifacts)
      .filter(([kind]) => AUDIO_KINDS.has(kind))
      .map(([id, url]) => ({
        id,
        label: STEM_LABELS[id] ?? id,
        url,
      }));
  }, [job]);

  const downloadArtifacts = job?.artifacts
    ? Object.entries(job.artifacts).filter(
        ([kind]) => kind !== "zip" && !isDiagnosticKind(kind)
      )
    : [];

  const guitarRadioOptions: { id: GuitarTrackChoice; label: string }[] = [
    { id: "none", label: "No guitar" },
    ...Array.from(GUITAR_TRACK_OPTION_IDS).map((id) => ({
      id,
      label: TRACK_OPTIONS[id].label,
    })),
  ];

  const sectionReady = duration !== null;
  const startBlocked =
    busy || Boolean(pickerError) || regionOverCap || (useRegion && !sectionReady);
  const isolateLabel = isolateStartLabel(busy, job?.status);
  const isolatePending =
    busy || job?.status === "pending" || job?.status === "running";

  return (
    <div>
      <h1>Isolate</h1>
      <p className="lede">
        {HOSTED_FILE_ONLY_NOTE} Pick labeled tracks (Demucs or BS-RoFormer for guitar), an
        optional section, and a processing mode.
      </p>
      <form className="stack" onSubmit={onSubmit}>
        <label className="field">
          Audio file
          <input
            type="file"
            accept=".mp3,.wav,.flac,.m4a,audio/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </label>

        <fieldset className="field">
          <legend>Tracks to separate</legend>
          <p className="hint">{TRACKS_PICKER_HELP}</p>
          <label className="field">
            <input
              type="checkbox"
              checked={vocalsInstrumentalOnly}
              onChange={(e) => setVocalsInstrumentalOnly(e.target.checked)}
            />{" "}
            {TRACK_OPTIONS[VOCALS_INSTRUMENTAL_OPTION_ID].label}
          </label>
          {!vocalsInstrumentalOnly && (
            <>
              <div className="stack" style={{ marginTop: "0.5rem" }}>
                {DEMUCS_STEM_CHECKBOX_IDS.map((id) => (
                  <label key={id} className="field">
                    <input
                      type="checkbox"
                      checked={demucsChecked[id]}
                      onChange={(e) => toggleDemucsStem(id, e.target.checked)}
                    />{" "}
                    {TRACK_OPTIONS[id].label}
                  </label>
                ))}
              </div>
              <fieldset className="field" style={{ marginTop: "0.75rem" }}>
                <legend>Guitar</legend>
                {guitarRadioOptions.map(({ id, label }) => (
                  <label key={id} className="field">
                    <input
                      type="radio"
                      name="guitar_track"
                      checked={guitarTrack === id}
                      onChange={() => setGuitarTrack(id)}
                    />{" "}
                    {label}
                  </label>
                ))}
              </fieldset>
            </>
          )}
          {pickerError ? <p className="error">{pickerError}</p> : null}
          {resolved.caveat ? <p className="hint">{resolved.caveat}</p> : null}
        </fieldset>

        <fieldset className="field" disabled={!file}>
          <legend>Section</legend>
          {!file ? (
            <span className="hint">Upload a file to choose a section or preview audio.</span>
          ) : (
            <>
              {duration !== null ? (
                <span className="hint">
                  Length: {formatTime(duration)} ({duration.toFixed(1)} s)
                </span>
              ) : (
                <span className="hint">
                  Length unknown. Section mode is available after the file length is
                  read. Until then the full file will be processed.
                </span>
              )}
              <label className="field">
                <input
                  type="checkbox"
                  checked={useRegion}
                  disabled={!sectionReady}
                  onChange={(e) => setUseRegion(e.target.checked)}
                />{" "}
                Isolate only a section
              </label>
              {objectUrl && (
                <audio
                  ref={audioRef}
                  src={objectUrl}
                  controls={!useRegion}
                  style={{ width: "100%", display: useRegion ? "none" : "block" }}
                  onLoadedMetadata={onAudioMeta}
                />
              )}
              {useRegion && duration !== null && objectUrl && (
                <>
                  <RegionPicker
                    audioUrl={objectUrl}
                    startSec={regionStart}
                    endSec={regionEnd}
                    minLengthSec={MIN_REGION_SEC}
                    onChange={onRegionChange}
                  />
                  {regionOverCap && selectedMode && (
                    <p className="error">
                      {selectedMode.label} allows up to {modeCap} s per job. Shorten the section
                      or choose a different mode.
                      <button
                        type="button"
                        className="secondary"
                        style={{ marginLeft: "0.5rem" }}
                        onClick={shortenEndToCap}
                      >
                        Shorten end to {modeCap} s
                      </button>
                    </p>
                  )}
                </>
              )}
              {useRegion && duration !== null && !objectUrl && (
                <>
                  <label className="field">
                    Start ({formatTime(regionStart)})
                    <input
                      type="range"
                      min={0}
                      max={duration}
                      step={1}
                      value={regionStart}
                      onChange={(e) => {
                        const next = Number(e.target.value);
                        setRegionStart(next);
                        if (regionEnd <= next + MIN_REGION_SEC) {
                          setRegionEnd(Math.min(duration, next + MIN_REGION_SEC));
                        }
                      }}
                    />
                  </label>
                  <label className="field">
                    End ({formatTime(regionEnd)})
                    <input
                      type="range"
                      min={MIN_REGION_SEC}
                      max={duration}
                      step={1}
                      value={regionEnd}
                      onChange={(e) => {
                        const next = Number(e.target.value);
                        setRegionEnd(next);
                        if (regionStart >= next - MIN_REGION_SEC) {
                          setRegionStart(Math.max(0, next - MIN_REGION_SEC));
                        }
                      }}
                    />
                  </label>
                  <span className="hint">
                    {regionLabel} ({(regionEnd - regionStart).toFixed(0)} s)
                  </span>
                  {regionOverCap && selectedMode && (
                    <p className="error">
                      {selectedMode.label} allows up to {modeCap} s per job. Shorten the section
                      or choose a different mode.
                      <button
                        type="button"
                        className="secondary"
                        style={{ marginLeft: "0.5rem" }}
                        onClick={shortenEndToCap}
                      >
                        Shorten end to {modeCap} s
                      </button>
                    </p>
                  )}
                </>
              )}
            </>
          )}
        </fieldset>

        <ProcessingModeSelect value={processingMode} onChange={setProcessingMode} />
        {trimNotice && <p className="hint">{trimNotice}</p>}
        <label className="field">
          Low-end restore (dB, 0 = off)
          <input
            type="number"
            min={0}
            max={6}
            step={1}
            value={lowEndRestoreDb}
            onChange={(e) => setLowEndRestoreDb(Number(e.target.value) || 0)}
          />
        </label>
        <label className="field checkbox">
          <input
            type="checkbox"
            checked={subBassDebleed}
            onChange={(e) => setSubBassDebleed(e.target.checked)}
          />
          Subtractive bass de-bleed (opt-in)
        </label>

        {error && <p className="error fade-in">{error}</p>}
        <button
          type="submit"
          disabled={isolateStartDisabled(startBlocked, job?.status)}
          aria-busy={isolatePending || undefined}
        >
          <StableLabel
            value={isolateLabel}
            options={["Start isolation", "Starting…", "Job running…"]}
            pending={isolatePending}
          />
        </button>
      </form>

      <div style={{ marginTop: "1.5rem" }}>
        {jobList.length > 0 && (
          <div className="results-card reveal" style={{ marginBottom: "1rem" }}>
            <h2 className="section-heading">Job queue</h2>
            <p className="hint">
              Jobs run one at a time. Queuing several songs means a long wait and high
              CPU/RAM use. This is not parallel Demucs.
            </p>
            <ul className="stack" style={{ listStyle: "none", padding: 0 }}>
              {jobList.map((j) => (
                <li key={j.id} style={{ marginBottom: "0.35rem" }}>
                  <button
                    type="button"
                    className="secondary"
                    onClick={() => {
                      stopWatch?.();
                      setJob(j);
                      if (!["succeeded", "failed", "cancelled"].includes(j.status)) {
                        const stop = api.watchJob(j.id, setJob);
                        setStopWatch(() => stop);
                      }
                    }}
                  >
                    {j.title ? `${j.title} · ${j.status}` : j.status}
                    {j.message ? ` · ${j.message}` : ""}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
        {clipLabel && job && (
          <p className="hint">
            <strong>Clip {clipLabel}</strong>
          </p>
        )}
        {job?.status !== "succeeded" && (
          <JobProgress
            job={job}
            onCancel={
              job && !["succeeded", "failed", "cancelled"].includes(job.status)
                ? () => void api.cancelJob(job.id).then(setJob)
                : undefined
            }
          />
        )}
      </div>

      {job?.status === "succeeded" && (
        <div className="results-card reveal" style={{ marginTop: "1rem" }}>
          <h2 className="section-heading">
            Your files
            {clipLabel && (
              <span className="hint" style={{ marginLeft: "0.5rem", fontWeight: "normal" }}>
                Clip {clipLabel}
              </span>
            )}
          </h2>
          <div className="artifacts">
            {job.artifacts.zip && (
              <a className="btn" href={job.artifacts.zip} download>
                Download all stems (ZIP)
              </a>
            )}
            {downloadArtifacts.map(([kind, url]) => (
              <a key={kind} className="btn secondary" href={url} download>
                Download {STEM_LABELS[kind] ?? kind}
              </a>
            ))}
          </div>
          {stems.length > 0 && (
            <StemMixer stems={stems} analysis={practice.analysis} tab={practice.tab} />
          )}
        </div>
      )}
    </div>
  );
}
