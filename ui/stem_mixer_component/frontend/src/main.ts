import {
  Streamlit,
  RenderData,
  Theme,
} from "streamlit-component-lib";
import "./style.css";
import {
  METRONOME_RATE_CHOICES,
  METRONOME_SOUND_LABELS,
  MetronomeConfig,
  MetronomeOptions,
  MetronomeSoundId,
  buildMetronomeAudioBuffer,
  coerceMetronomeFollow,
  coerceMetronomeOptions,
  coerceMetronomeRate,
  coerceMetronomeSound,
  metronomeClickTimes,
  metronomePeaksFromTimes,
} from "./metronomeClicks";
// @soundtouchjs/audio-worklet@2.1.1 — MPL-2.0 (not GPL Rubber Band).
import type { SoundTouchNode } from "@soundtouchjs/audio-worklet";
import soundTouchProcessorUrl from "@soundtouchjs/audio-worklet/processor?url";
import {
  PRACTICE_KEY_DEFAULT,
  PRACTICE_RATE_DEFAULT,
  PRACTICE_RATE_STEP,
  clampPracticeRate,
  clampPracticeSemitones,
  formatPracticeKeyLabel,
  formatPracticeSpeedLabel,
  loadPracticeSettings,
  savePracticeSettings,
  type PracticeSettings,
} from "./practice";
import {
  PAN_DEFAULT,
  PAN_MAX,
  PAN_MIN,
  clampPan,
  createStemPanner,
  formatPan,
  panSpoken,
  panToUnit,
  snapPanToCenter,
} from "./pan";
import {
  advanceChordIndex,
  capoDiagramSymbol,
  chartTimeSeconds,
  chordAt,
  CHORD_DIAGRAM_LAYOUT,
  chordShape,
  displaySymbol,
  outputLatencyMs,
  scrollLeftToCenter,
  CAPO_PRESETS,
  chordDifficulty,
  formatBassTab,
  listenCents,
  replaceChordSymbol,
  replaceLyricLine,
  replaceMeasureText,
  replaceSectionLabel,
  spellChordSymbol,
  clampLoop,
  formatChordSymbol,
  lyricLineAt,
  measureAt,
  sectionAt,
  type AnalysisData,
  type ChordLevel,
  type Section,
  type TabData,
} from "./analysis";

let practiceSheetOpen: boolean | null = null;
const chordSymbolEdits = new Map<number, string>();
let chordEditKey = "";
let capoFret = 0;
let chordLevel: ChordLevel = "medium";
let chordFollowIndex = 0;
let chordActiveChip: HTMLElement | null = null;
let chordUserScrollAt = 0;
let chordProgrammaticScroll = false;
let lastSidebarChord = "";

function adoptAnalysis(next: AnalysisData | null): AnalysisData | null {
  const key = (next?.chords ?? []).map((chord) => String(chord.start)).join(",");
  if (key !== chordEditKey) {
    chordEditKey = key;
    chordSymbolEdits.clear();
    capoFret = 0;
    chordFollowIndex = 0;
    chordActiveChip = null;
  }
  if (next?.chords) {
    for (const [index, symbol] of chordSymbolEdits) {
      const chord = next.chords[index];
      if (chord) chord.symbol = symbol;
    }
  }
  return next;
}

const MIXER_MENUS: ReadonlyArray<readonly [string, string]> = [
  ["practice-speed-popover", "btn-speed-key"],
  ["mix-menu", "btn-mix"],
];

function dismissMixerMenus(target: Node | null): void {
  for (const [panelId, buttonId] of MIXER_MENUS) {
    const panel = document.getElementById(panelId);
    const button = document.getElementById(buttonId);
    if (!panel || !button || panel.hidden) continue;
    if (target && (panel.contains(target) || button.contains(target))) continue;
    panel.hidden = true;
    button.setAttribute("aria-expanded", "false");
  }
  syncMixerMenuBackdrop();
}

function syncMixerMenuBackdrop(): void {
  const backdrop = document.getElementById("mixer-menu-backdrop");
  if (!backdrop) return;
  const open = MIXER_MENUS.some(([panelId]) => {
    const panel = document.getElementById(panelId);
    return Boolean(panel && !panel.hidden);
  });
  backdrop.hidden = !open;
}

function onSpeedPopoverEscape(event: KeyboardEvent): void {
  if (event.key !== "Escape") return;
  const popover = document.getElementById("practice-speed-popover");
  const button = document.getElementById("btn-speed-key");
  const mix = document.getElementById("mix-menu");
  const mixButton = document.getElementById("btn-mix");
  const speedOpen = Boolean(popover && button && !popover.hidden);
  const mixOpen = Boolean(mix && mixButton && !mix.hidden);
  if (!speedOpen && !mixOpen) return;
  event.preventDefault();
  dismissMixerMenus(null);
  (mixOpen ? mixButton : button)?.focus();
}

function onMixerMenuPointerDown(event: PointerEvent): void {
  dismissMixerMenus(event.target as Node | null);
}

function onMixerMenuWindowBlur(): void {
  dismissMixerMenus(null);
}

const DB_MIN = -25;
const DB_MAX = 25;
const DB_DEFAULT = 0;

function clampDb(db: number): number {
  return Math.max(DB_MIN, Math.min(DB_MAX, db));
}

type StemInfo = {
  id: string;
  label: string;
  url: string;
  downloadUrl?: string;
  downloadFilename?: string;
  peaks?: number[];
  hint?: string;
  /** Parent stem id when this stem came from "Break down a stem" (e.g. `other`). */
  group?: string;
  groupLabel?: string;
  /** Row name inside its group, without the "(from Parent)" tail. */
  shortLabel?: string;
};

type MixerState = {
  volumesDb: Record<string, number>;
  /** Whole steps -100 (left) .. 0 (center) .. 100 (right). Law in ./pan.ts. */
  pan: Record<string, number>;
  muted: Record<string, boolean>;
  soloed: Record<string, boolean>;
  masterVolumeDb: number;
};

function dbToLinear(db: number): number {
  if (db <= DB_MIN) return 0;
  return Math.pow(10, db / 20);
}

function effectiveGains(
  stemIds: string[],
  state: MixerState
): Record<string, number> {
  const anySolo = stemIds.some((id) => state.soloed[id]);
  const masterLin = dbToLinear(
    Math.max(DB_MIN, Math.min(DB_MAX, state.masterVolumeDb ?? DB_DEFAULT))
  );
  const gains: Record<string, number> = {};
  for (const id of stemIds) {
    const db = Math.max(
      DB_MIN,
      Math.min(DB_MAX, state.volumesDb[id] ?? DB_DEFAULT)
    );
    const audible = anySolo ? !!state.soloed[id] : !state.muted[id];
    gains[id] = audible ? dbToLinear(db) * masterLin : 0;
  }
  return gains;
}

/** Tiny silent wav so the OS treats the mixer as active media. */
const KEEP_ALIVE_SRC =
  "data:audio/wav;base64,UklGRnoAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoAAAAAAAAAAAAAAAA=";

let keepAliveEl: HTMLAudioElement | null = null;
let keepAliveResume: (() => void) | null = null;

function startKeepAliveAudio(onPlay?: () => void): void {
  if (typeof document === "undefined") return;
  keepAliveResume = onPlay ?? keepAliveResume;
  if (!keepAliveEl) {
    const el = document.createElement("audio");
    el.src = KEEP_ALIVE_SRC;
    el.loop = true;
    el.muted = true;
    el.setAttribute("playsinline", "true");
    el.preload = "auto";
    el.setAttribute("aria-hidden", "true");
    el.style.display = "none";
    el.addEventListener("play", () => {
      keepAliveResume?.();
    });
    document.body.appendChild(el);
    keepAliveEl = el;
  }
  void keepAliveEl.play()?.catch(() => {
    /* gesture already used for stems; retry on next wake */
  });
}

function stopKeepAliveAudio(): void {
  if (!keepAliveEl) return;
  try {
    keepAliveEl.pause();
    keepAliveEl.currentTime = 0;
  } catch {
    /* element already gone */
  }
}

function claimAudioSessionPlayback(): void {
  const nav = navigator as Navigator & { audioSession?: { type: string } };
  try {
    if (nav.audioSession) nav.audioSession.type = "playback";
  } catch {
    /* Safari/WKWebView only */
  }
}

class StemMixerEngine {
  private ctx: AudioContext | null = null;
  private buffers = new Map<string, AudioBuffer>();
  private gains = new Map<string, GainNode>();
  /** Per-stem pan (stem gain -> panner -> master). Absent where the browser has none. */
  private panners = new Map<string, StereoPannerNode>();
  private sources = new Map<string, AudioBufferSourceNode>();
  private masterGain: GainNode | null = null;
  private limiter: DynamicsCompressorNode | null = null;
  private lowShelf: BiquadFilterNode | null = null;
  private highShelf: BiquadFilterNode | null = null;
  private stemTrims: Record<string, { start: number; end: number }> = {};
  /** One shared SoundTouch on the master bus — stems stay phase-aligned, lower CPU. */
  private soundTouch: SoundTouchNode | null = null;
  private soundTouchFailed = false;
  private practiceRate = PRACTICE_RATE_DEFAULT;
  private practiceSemitones = PRACTICE_KEY_DEFAULT;
  private listenCents = 0;
  private streamDest: MediaStreamAudioDestinationNode | null = null;
  private outputEl: HTMLAudioElement | null = null;
  private stemIds: string[] = [];
  private playing = false;
  private startedAt = 0;
  private offset = 0;
  private duration = 0;
  private loopStart: number | null = null;
  private loopEnd: number | null = null;
  private countInBars = 0;
  private countInBpm = 120;
  private countInBeatsPerMeasure = 4;
  private countInOptions: MetronomeOptions = coerceMetronomeOptions({
    accent: true,
    rate: 1,
    sound: "classic",
    follow: "smart",
  });
  private countInSource: AudioBufferSourceNode | null = null;
  private countInUntil = 0;
  private lastGains: Record<string, number> = {};
  private lastPans: Record<string, number> = {};
  private onTimeUpdate: ((t: number, dur: number, playing: boolean) => void) | null =
    null;
  private onSoftPause: (() => void) | null = null;
  private raf = 0;
  private wakeHooked = false;
  /** User pressed Play. Window blur and sleep must not clear this. */
  private keepPlaying = false;
  private resumeInFlight = false;
  private resumeAttemptAt = 0;

  setTimeCallback(
    cb: (t: number, dur: number, playing: boolean) => void
  ): void {
    this.onTimeUpdate = cb;
  }

  setSoftPauseCallback(cb: () => void): void {
    this.onSoftPause = cb;
  }

  /**
   * Keep a Play press alive across window switches and sleep.
   * The browser may suspend the context; that is not a user pause.
   */
  installWakeHooks(): void {
    if (this.wakeHooked || typeof window === "undefined") return;
    this.wakeHooked = true;
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible") {
        void this.handleWake();
      }
    });
    window.addEventListener("pageshow", () => {
      void this.handleWake();
    });
    window.addEventListener("focus", () => {
      void this.handleWake();
    });
  }

  getDuration(): number {
    return this.duration;
  }

  getBuffer(id: string): AudioBuffer | undefined {
    return this.buffers.get(id);
  }

  isPlaying(): boolean {
    return this.playing;
  }

  isKeepPlaying(): boolean {
    return this.keepPlaying;
  }

  chartLatencyMs(userOffsetMs = 0): number {
    return outputLatencyMs(this.ctx, userOffsetMs);
  }

  currentTime(): number {
    if (!this.ctx) return this.offset;
    if (!this.playing) return this.offset;
    // Count-in schedules sources in the future. Before that start line the
    // transport must hold at the seek/loop point, not run backwards from it.
    if (this.ctx.currentTime < this.startedAt) return this.offset;
    const rate = this.effectivePlaybackRate();
    const raw = this.offset + (this.ctx.currentTime - this.startedAt) * rate;
    if (raw <= 0) return 0;
    if (this.hasLoop()) {
      const start = this.loopStart!;
      const end = this.loopEnd!;
      const span = end - start;
      if (this.offset >= start && this.offset < end && raw >= end && span > 0) {
        return start + ((raw - start) % span);
      }
    }
    return Math.min(this.duration, raw);
  }

  hasLoop(): boolean {
    return (
      this.loopStart !== null &&
      this.loopEnd !== null &&
      this.loopEnd - this.loopStart >= 0.5
    );
  }

  getLoopRegion(): { start: number; end: number } | null {
    return this.hasLoop()
      ? { start: this.loopStart!, end: this.loopEnd! }
      : null;
  }

  /**
   * Set or clear the A/B loop. Invalid/short regions clear it instead of
   * starting an accidental one-beat stutter. Active sources are updated live;
   * the transport jumps to the loop start if it is currently outside.
   */
  setLoopRegion(start: number | null, end: number | null): void {
    if (
      start === null ||
      end === null ||
      !Number.isFinite(start) ||
      !Number.isFinite(end) ||
      this.duration <= 0
    ) {
      this.clearLoop();
      return;
    }
    const s = Math.max(0, Math.min(this.duration, start));
    const e = Math.max(0, Math.min(this.duration, end));
    if (!(e - s >= 0.5)) {
      this.clearLoop();
      return;
    }
    this.loopStart = s;
    this.loopEnd = e;
    for (const src of this.sources.values()) {
      try {
        src.loop = true;
        src.loopStart = s;
        src.loopEnd = e;
      } catch {
        /* source already stopped */
      }
    }
    if (this.playing && this.ctx) {
      const now = this.currentTime();
      if (now < s || now >= e) {
        this.offset = s;
        this.startedAt = this.ctx.currentTime;
        void this.seek(s);
      }
    }
  }

  clearLoop(): void {
    this.loopStart = null;
    this.loopEnd = null;
    for (const src of this.sources.values()) {
      try {
        src.loop = false;
      } catch {
        /* source already stopped */
      }
    }
  }

  setCountInBars(bars: number): void {
    const n = Number(bars);
    this.countInBars = Number.isFinite(n) ? Math.max(0, Math.min(2, Math.round(n))) : 0;
  }

  getCountInBars(): number {
    return this.countInBars;
  }

  setCountInGrid(
    bpm: number | null | undefined,
    beatsPerMeasure: number | null | undefined,
    options?: Partial<MetronomeOptions>
  ): void {
    const tempo = Number(bpm);
    this.countInBpm = Number.isFinite(tempo) && tempo > 0 ? tempo : 120;
    const beats = Number(beatsPerMeasure);
    this.countInBeatsPerMeasure =
      Number.isFinite(beats) && beats > 0 ? Math.max(1, Math.round(beats)) : 4;
    if (options) this.countInOptions = coerceMetronomeOptions(options);
  }

  countInRemaining(): number {
    if (!this.playing || !this.ctx || this.countInUntil <= 0) return 0;
    return Math.max(0, this.countInUntil - this.ctx.currentTime);
  }

  private stopCountIn(): void {
    this.countInUntil = 0;
    const src = this.countInSource;
    this.countInSource = null;
    if (!src) return;
    try {
      src.onended = null;
      src.stop();
    } catch {
      /* already stopped */
    }
    try {
      src.disconnect();
    } catch {
      /* already disconnected */
    }
  }

  /** Schedule one bar of clicks before the stems start; returns its duration. */
  private startCountIn(): number {
    this.stopCountIn();
    if (!this.ctx || !this.masterGain || this.countInBars <= 0) return 0;
    const beatSec = 60 / Math.max(30, this.countInBpm);
    const totalBeats = this.countInBars * this.countInBeatsPerMeasure;
    const durationSec = beatSec * totalBeats;
    const times = Array.from({ length: totalBeats }, (_, i) => i * beatSec);
    // A count-in is one click per beat. The metronome's subdivision choice
    // applies after the song starts, not to the lead-in bar.
    const options = coerceMetronomeOptions({
      ...this.countInOptions,
      rate: 1,
      follow: "steady",
    });
    const config: MetronomeConfig = {
      times1x: times,
      steadyTimes1x: times,
      refSec: 0,
      beatsPerMeasure: this.countInBeatsPerMeasure,
      durationSec: Math.max(0.05, durationSec),
      options,
    };
    const buffer = buildMetronomeAudioBuffer(this.ctx, config, options);
    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(this.masterGain);
    const startAt = this.ctx.currentTime + 0.03;
    src.start(startAt);
    src.onended = () => {
      if (this.countInSource === src) this.countInSource = null;
    };
    this.countInSource = src;
    this.countInUntil = startAt + durationSec;
    return durationSec;
  }

  getPracticeSettings(): PracticeSettings {
    return { rate: this.practiceRate, semitones: this.practiceSemitones };
  }

  /** Listen-only pitch. Export still uses the Key slider. */
  setListenHertz(hz: number | null): void {
    this.listenCents = hz == null ? 0 : listenCents(hz);
    this.applyPracticeToGraph();
  }

  /** Listen-only shelves. They do not change the saved mix. */
  setShelves(lowDb: number, highDb: number): void {
    if (this.lowShelf) this.lowShelf.gain.value = lowDb;
    if (this.highShelf) this.highShelf.gain.value = highDb;
  }

  setStemTrims(trims: Record<string, { start?: number; end?: number }> | null | undefined): void {
    this.stemTrims = {};
    if (!trims) return;
    for (const [id, value] of Object.entries(trims)) {
      const start = Number(value?.start);
      const end = Number(value?.end);
      if (Number.isFinite(start) && Number.isFinite(end) && end > start) {
        this.stemTrims[id] = { start, end };
      }
    }
  }

  isPracticeAvailable(): boolean {
    return this.soundTouch !== null && !this.soundTouchFailed;
  }

  /** Wall-clock→audio rate used for stems/picture when SoundTouch is active. */
  getEffectivePlaybackRate(): number {
    return this.soundTouch ? this.practiceRate : PRACTICE_RATE_DEFAULT;
  }

  private effectivePlaybackRate(): number {
    return this.getEffectivePlaybackRate();
  }

  /** Create a context for decode only. Never resume — pywebview blocks that without a click. */
  createContextForDecode(): AudioContext {
    if (!this.ctx || this.ctx.state === "closed") {
      this.ctx = new AudioContext();
      this.teardownSoundTouch();
      this.masterGain = null;
      this.limiter = null;
      this.lowShelf = null;
      this.highShelf = null;
    }
    this.ensureMasterBus();
    this.bindContextState();
    return this.ctx;
  }

  async ensureContext(): Promise<AudioContext> {
    if (!this.ctx || this.ctx.state === "closed") {
      this.rebuildGraphKeepingBuffers();
    }
    const state = this.ctx!.state as string;
    if (state === "suspended" || state === "interrupted") {
      try {
        await this.ctx!.resume();
      } catch {
        // While Play is held, do not wipe sources — resume() often fails without a gesture.
        if (!this.keepPlaying) this.rebuildGraphKeepingBuffers();
      }
    }
    if (this.ctx!.state === "closed") {
      this.rebuildGraphKeepingBuffers();
    }
    this.ensureMasterBus();
    this.bindContextState();
    return this.ctx!;
  }

  /**
   * Master gain → [SoundTouch] → limiter → MediaStreamDestination → HTMLAudioElement.
   * Routing through a media element keeps WKWebView/WebView2 from treating
   * the mix as silent Web Audio that can suspend on window blur.
   */
  private ensureMasterBus(): void {
    if (!this.ctx) return;
    if (this.masterGain && this.limiter) return;
    this.masterGain = this.ctx.createGain();
    this.masterGain.gain.value = 1;
    this.limiter = this.ctx.createDynamicsCompressor();
    this.lowShelf = this.ctx.createBiquadFilter();
    this.lowShelf.type = "lowshelf";
    this.lowShelf.frequency.value = 200;
    this.lowShelf.gain.value = 0;
    this.highShelf = this.ctx.createBiquadFilter();
    this.highShelf.type = "highshelf";
    this.highShelf.frequency.value = 4000;
    this.highShelf.gain.value = 0;
    // Soft ceiling near −1 dBFS true-peak proxy (matches mixer.py TRUE_PEAK_CEILING).
    this.limiter.threshold.value = -1;
    this.limiter.knee.value = 0;
    this.limiter.ratio.value = 20;
    this.limiter.attack.value = 0.003;
    this.limiter.release.value = 0.1;
    this.masterGain.connect(this.limiter);
    this.limiter.connect(this.lowShelf);
    this.lowShelf.connect(this.highShelf);
    if (typeof this.ctx.createMediaStreamDestination === "function") {
      this.streamDest = this.ctx.createMediaStreamDestination();
      this.highShelf.connect(this.streamDest);
      this.ensureOutputElement();
    } else {
      this.highShelf.connect(this.ctx.destination);
    }
  }

  private teardownSoundTouch(): void {
    if (this.soundTouch) {
      try {
        this.soundTouch.disconnect();
      } catch {
        /* */
      }
      this.soundTouch = null;
    }
  }

  /**
   * Insert one SoundTouch node on the master bus after stem gains.
   * Failure leaves the graph at 1× / 0 semitones (no chipmunk fallback).
   */
  async ensureSoundTouch(): Promise<boolean> {
    if (this.soundTouchFailed) return false;
    if (this.soundTouch) return true;
    if (!this.ctx) return false;
    this.ensureMasterBus();
    if (!this.masterGain || !this.limiter) return false;
    try {
      const { SoundTouchNode: ST } = await import("@soundtouchjs/audio-worklet");
      await ST.register(this.ctx, soundTouchProcessorUrl);
      const st = new ST({ context: this.ctx });
      this.masterGain.disconnect();
      this.masterGain.connect(st);
      st.connect(this.limiter);
      this.soundTouch = st;
      this.applyPracticeToGraph();
      return true;
    } catch {
      this.soundTouchFailed = true;
      this.practiceRate = PRACTICE_RATE_DEFAULT;
      this.practiceSemitones = PRACTICE_KEY_DEFAULT;
      try {
        this.masterGain.disconnect();
        this.masterGain.connect(this.limiter);
      } catch {
        /* */
      }
      return false;
    }
  }

  private applyPracticeToGraph(): void {
    const rate = this.effectivePlaybackRate();
    const semis = this.soundTouch ? this.practiceSemitones : PRACTICE_KEY_DEFAULT;
    if (this.soundTouch) {
      this.soundTouch.playbackRate.value = rate;
      this.soundTouch.pitch.value = 1;
      this.soundTouch.pitchSemitones.value = semis + this.listenCents / 100;
    }
    for (const src of this.sources.values()) {
      try {
        src.playbackRate.value = rate;
      } catch {
        /* */
      }
    }
  }

  async setPracticeRate(rate: number): Promise<{ ok: boolean; message?: string }> {
    const next = clampPracticeRate(rate);
    if (this.soundTouchFailed && next !== PRACTICE_RATE_DEFAULT) {
      this.practiceRate = PRACTICE_RATE_DEFAULT;
      this.applyPracticeToGraph();
      return {
        ok: false,
        message: "Speed/Key unavailable — practice worklet failed to load.",
      };
    }
    if (this.playing && this.ctx) {
      this.offset = this.currentTime();
      this.startedAt = this.ctx.currentTime;
    }
    this.practiceRate = next;
    if (this.ctx) {
      const ready = await this.ensureSoundTouch();
      if (!ready && next !== PRACTICE_RATE_DEFAULT) {
        this.practiceRate = PRACTICE_RATE_DEFAULT;
        this.applyPracticeToGraph();
        return {
          ok: false,
          message: "Speed/Key unavailable — practice worklet failed to load.",
        };
      }
    }
    this.applyPracticeToGraph();
    return { ok: true };
  }

  async setPracticeSemitones(
    semitones: number
  ): Promise<{ ok: boolean; message?: string }> {
    const next = clampPracticeSemitones(semitones);
    if (this.soundTouchFailed && next !== PRACTICE_KEY_DEFAULT) {
      this.practiceSemitones = PRACTICE_KEY_DEFAULT;
      this.applyPracticeToGraph();
      return {
        ok: false,
        message: "Speed/Key unavailable — practice worklet failed to load.",
      };
    }
    this.practiceSemitones = next;
    if (this.ctx) {
      const ready = await this.ensureSoundTouch();
      if (!ready && next !== PRACTICE_KEY_DEFAULT) {
        this.practiceSemitones = PRACTICE_KEY_DEFAULT;
        this.applyPracticeToGraph();
        return {
          ok: false,
          message: "Speed/Key unavailable — practice worklet failed to load.",
        };
      }
    }
    this.applyPracticeToGraph();
    return { ok: true };
  }

  async resetPractice(): Promise<{ ok: boolean; message?: string }> {
    const rate = await this.setPracticeRate(PRACTICE_RATE_DEFAULT);
    const key = await this.setPracticeSemitones(PRACTICE_KEY_DEFAULT);
    if (!rate.ok) return rate;
    if (!key.ok) return key;
    return { ok: true };
  }

  private ensureOutputElement(): void {
    if (!this.streamDest || typeof document === "undefined") return;
    if (!this.outputEl) {
      const el = document.createElement("audio");
      el.setAttribute("playsinline", "true");
      el.preload = "auto";
      el.setAttribute("aria-hidden", "true");
      el.style.display = "none";
      el.addEventListener("play", () => {
        void this.continueAfterInterrupt();
      });
      document.body.appendChild(el);
      this.outputEl = el;
    }
    if (this.outputEl.srcObject !== this.streamDest.stream) {
      this.outputEl.srcObject = this.streamDest.stream;
    }
  }

  private async startStreamOutput(): Promise<void> {
    claimAudioSessionPlayback();
    this.ensureMasterBus();
    this.ensureOutputElement();
    if (this.outputEl) {
      try {
        await this.outputEl.play();
        return;
      } catch {
        /* fall through to silent keep-alive */
      }
    }
    startKeepAliveAudio(() => {
      void this.continueAfterInterrupt();
    });
  }

  private stopStreamOutput(): void {
    if (this.outputEl) {
      try {
        this.outputEl.pause();
      } catch {
        /* */
      }
    }
    stopKeepAliveAudio();
  }

  private bindContextState(): void {
    if (!this.ctx) return;
    this.ctx.onstatechange = () => {
      const st = this.ctx?.state as string | undefined;
      if (st !== "interrupted" && st !== "closed" && st !== "suspended") return;
      if (!this.keepPlaying) return;
      const now = performance.now();
      if (now - this.resumeAttemptAt < 500) return;
      this.resumeAttemptAt = now;
      void this.continueAfterInterrupt();
    };
  }

  /** Recreate AudioContext + bus; keep decoded buffers (wake recovery). */
  private rebuildGraphKeepingBuffers(): void {
    const savedOffset = this.playing && this.ctx ? this.currentTime() : this.offset;
    for (const src of this.sources.values()) {
      try {
        src.onended = null;
        src.stop();
      } catch {
        /* */
      }
      try {
        src.disconnect();
      } catch {
        /* */
      }
    }
    this.sources.clear();
    this.playing = false;
    cancelAnimationFrame(this.raf);
    for (const node of [...this.gains.values(), ...this.panners.values()]) {
      try {
        node.disconnect();
      } catch {
        /* */
      }
    }
    this.gains.clear();
    this.panners.clear();
    this.teardownSoundTouch();
    this.soundTouchFailed = false;
    this.masterGain = null;
    this.limiter = null;
    this.lowShelf = null;
    this.highShelf = null;
    this.streamDest = null;
    const old = this.ctx;
    this.ctx = null;
    if (old) {
      try {
        void old.close();
      } catch {
        /* */
      }
    }
    this.ctx = new AudioContext();
    this.ensureMasterBus();
    this.bindContextState();
    for (const id of this.stemIds) {
      if (!this.buffers.has(id)) continue;
      this.addStemBus(this.ctx, id);
    }
    this.offset = savedOffset;
    this.applyGains(this.lastGains);
    this.applyPans(this.lastPans);
  }

  private async softPauseFromInterrupt(): Promise<void> {
    // Blur/sleep must not clear Play. Retry when the context can run again.
    await this.continueAfterInterrupt();
  }

  private async contextLooksZombie(): Promise<boolean> {
    if (!this.ctx || this.ctx.state !== "running") return false;
    const t0 = this.ctx.currentTime;
    await new Promise((r) => window.setTimeout(r, 150));
    if (!this.ctx || this.ctx.state !== "running") return false;
    return this.ctx.currentTime === t0;
  }

  async handleWake(): Promise<void> {
    if (!this.ctx) return;
    const dead = this.ctx.state === "closed" || (await this.contextLooksZombie());
    if (dead) this.rebuildGraphKeepingBuffers();
    if (this.keepPlaying) await this.continueAfterInterrupt();
  }

  /** Resume the same Play press after the browser suspends the context. */
  private async continueAfterInterrupt(): Promise<void> {
    if (!this.keepPlaying || this.resumeInFlight) return;
    this.resumeInFlight = true;
    try {
      if (this.playing && this.ctx) this.offset = this.currentTime();
      if (this.ctx?.state === "closed") this.rebuildGraphKeepingBuffers();
      await this.ensureContext();
      await this.ensureSoundTouch();
      if (!this.ctx || (this.ctx.state as string) !== "running") return;
      await this.startStreamOutput();
      if (this.playing && this.sources.size > 0) return;
      if (this.offset >= this.duration) this.offset = 0;
      this.startSources(this.offset);
    } catch {
      /* keepPlaying stays set; the next focus or wake retries */
    } finally {
      this.resumeInFlight = false;
    }
  }

  async loadStems(
    stems: StemInfo[],
    onProgress: (loaded: number, total: number, message: string) => void,
    liveBuffers?: Record<string, (ctx: AudioContext, durationSec: number) => AudioBuffer>
  ): Promise<{ errors: string[] }> {
    await this.stopSources(false);
    for (const node of [...this.gains.values(), ...this.panners.values()]) node.disconnect();
    this.buffers.clear();
    this.gains.clear();
    this.panners.clear();
    this.sources.clear();
    this.stemIds = stems.map((s) => s.id);
    this.offset = 0;
    this.playing = false;
    this.keepPlaying = false;
    this.duration = 0;
    this.clearLoop();
    this.stopCountIn();
    stopKeepAliveAudio();

    // Decode without starting playback. Create context but don't resume until Play —
    // pywebview/WKWebView leaves output silent if resume() runs without a user gesture.
    const ctx = this.createContextForDecode();
    const errors: string[] = [];
    let loaded = 0;
    const total = stems.length;
    const deferred: StemInfo[] = [];

    for (const stem of stems) {
      if (liveBuffers?.[stem.id]) {
        deferred.push(stem);
        continue;
      }
      onProgress(loaded, total, `Loading ${stem.label}…`);
      try {
        const res = await fetch(stem.url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const arr = await res.arrayBuffer();
        const buf = await ctx.decodeAudioData(arr.slice(0));
        this.buffers.set(stem.id, buf);
        this.duration = Math.max(this.duration, buf.duration);
        this.addStemBus(ctx, stem.id);
      } catch (e) {
        errors.push(`${stem.label}: ${e instanceof Error ? e.message : String(e)}`);
      }
      loaded += 1;
      onProgress(loaded, total, `Loaded ${loaded}/${total}`);
    }

    for (const stem of deferred) {
      onProgress(loaded, total, `Loading ${stem.label}…`);
      try {
        const synth = liveBuffers![stem.id]!;
        const buf = synth(ctx, this.duration);
        this.buffers.set(stem.id, buf);
        this.duration = Math.max(this.duration, buf.duration);
        this.addStemBus(ctx, stem.id);
      } catch (e) {
        errors.push(`${stem.label}: ${e instanceof Error ? e.message : String(e)}`);
      }
      loaded += 1;
      onProgress(loaded, total, `Loaded ${loaded}/${total}`);
    }

    this.applyGains(
      effectiveGains(this.stemIds, {
        volumesDb: Object.fromEntries(this.stemIds.map((id) => [id, DB_DEFAULT])),
        pan: {},
        muted: {},
        soloed: {},
        masterVolumeDb: DB_DEFAULT,
      })
    );
    this.tick();
    return { errors };
  }

  /** Hot-swap one stem buffer without stopping the rest of the mix. */
  replaceStemBuffer(id: string, buf: AudioBuffer): void {
    this.buffers.set(id, buf);
    this.duration = 0;
    for (const b of this.buffers.values()) {
      this.duration = Math.max(this.duration, b.duration);
    }
    if (!this.ctx || !this.masterGain) return;
    if (!this.gains.has(id)) {
      this.addStemBus(this.ctx, id);
      if (!this.stemIds.includes(id)) this.stemIds.push(id);
    }
    if (!this.playing) {
      this.tick();
      return;
    }
    const offset = this.currentTime();
    const old = this.sources.get(id);
    if (old) {
      try {
        old.onended = null;
        old.stop();
      } catch {
        /* */
      }
      try {
        old.disconnect();
      } catch {
        /* */
      }
      this.sources.delete(id);
    }
    const gain = this.gains.get(id);
    if (gain && offset < buf.duration) {
      const src = this.ctx.createBufferSource();
      src.buffer = buf;
      src.playbackRate.value = this.effectivePlaybackRate();
      src.connect(gain);
      src.start(this.ctx.currentTime, offset);
      this.sources.set(id, src);
    }
    this.tick();
  }

  /** One stem's bus: source -> gain -> panner -> master. Law: see ./pan.ts. */
  private addStemBus(ctx: AudioContext, id: string): void {
    const gain = ctx.createGain();
    const panner = createStemPanner(ctx);
    if (panner) {
      gain.connect(panner);
      panner.connect(this.masterGain!);
      panner.pan.value = panToUnit(this.lastPans[id] ?? PAN_DEFAULT);
      this.panners.set(id, panner);
    } else {
      gain.connect(this.masterGain!);
    }
    this.gains.set(id, gain);
  }

  /** Pan per stem in whole steps (-100..100). Smoothed so a drag does not zipper. */
  applyPans(pans: Record<string, number>): void {
    this.lastPans = { ...pans };
    for (const id of this.stemIds) {
      const node = this.panners.get(id);
      if (!node || !this.ctx) continue;
      node.pan.setTargetAtTime(panToUnit(pans[id] ?? PAN_DEFAULT), this.ctx.currentTime, 0.015);
    }
  }

  applyGains(gains: Record<string, number>): void {
    this.lastGains = { ...gains };
    for (const id of this.stemIds) {
      const node = this.gains.get(id);
      if (!node || !this.ctx) continue;
      const g = gains[id] ?? 0;
      node.gain.setTargetAtTime(g, this.ctx.currentTime, 0.015);
    }
  }

  private async stopSources(preserveOffset: boolean): Promise<void> {
    if (preserveOffset && this.playing && this.ctx) {
      this.offset = this.currentTime();
    }
    this.stopCountIn();
    for (const src of this.sources.values()) {
      try {
        src.onended = null;
        src.stop();
      } catch {
        /* already stopped */
      }
      try {
        src.disconnect();
      } catch {
        /* */
      }
    }
    this.sources.clear();
    this.playing = false;
    cancelAnimationFrame(this.raf);
  }

  private startSources(offset: number, delaySec = 0): void {
    if (!this.ctx) return;
    const startAt = this.ctx.currentTime + 0.03 + Math.max(0, delaySec);
    this.startedAt = startAt;
    this.offset = offset;
    this.playing = true;
    const rate = this.effectivePlaybackRate();
    const looped = this.hasLoop();

    for (const id of this.stemIds) {
      const buf = this.buffers.get(id);
      const gain = this.gains.get(id);
      if (!buf || !gain) continue;
      if (offset >= buf.duration) continue;
      const src = this.ctx.createBufferSource();
      src.buffer = buf;
      src.playbackRate.value = rate;
      if (looped) {
        src.loop = true;
        src.loopStart = this.loopStart!;
        src.loopEnd = this.loopEnd!;
      }
      src.connect(gain);
      const trim = this.stemTrims[id];
      if (trim) {
        const playOffset = Math.max(offset, trim.start);
        if (playOffset >= trim.end || playOffset >= buf.duration) {
          try {
            src.disconnect();
          } catch {
            /* */
          }
          continue;
        }
        src.start(startAt, playOffset, trim.end - playOffset);
      } else {
        src.start(startAt, offset);
      }
      src.onended = () => {
        /* individual stem end — transport ends when timeline hits duration */
      };
      this.sources.set(id, src);
    }
    this.applyPracticeToGraph();
    this.tick();
  }

  async play(): Promise<void> {
    this.keepPlaying = true;
    await this.ensureContext();
    await this.ensureSoundTouch();
    // After sleep, playing may stay true with dead sources — force restart.
    if (this.playing) {
      const healthy =
        this.ctx?.state === "running" && this.sources.size > 0;
      if (healthy) {
        await this.startStreamOutput();
        return;
      }
      await this.stopSources(true);
    }
    if (this.offset >= this.duration) this.offset = 0;
    if (this.hasLoop() && (this.offset < this.loopStart! || this.offset >= this.loopEnd!)) {
      this.offset = this.loopStart!;
    }
    const countInSec = this.startCountIn();
    this.startSources(this.offset, countInSec);
    await this.startStreamOutput();
  }

  async pause(): Promise<void> {
    this.keepPlaying = false;
    this.stopStreamOutput();
    await this.stopSources(true);
    this.tick();
  }

  async seek(seconds: number): Promise<void> {
    const wasPlaying = this.playing;
    let t = Math.max(0, Math.min(this.duration, seconds));
    if (wasPlaying && this.hasLoop() && (t < this.loopStart! || t >= this.loopEnd!)) {
      // Native BufferSource looping only wraps once a source starts inside the
      // region. Keep the transport and the audio on the same timeline.
      t = this.loopStart!;
    }
    await this.stopSources(false);
    this.offset = t;
    if (wasPlaying) {
      this.startSources(this.offset);
    } else {
      this.tick();
    }
  }

  async restart(): Promise<void> {
    await this.seek(0);
    await this.play();
  }

  async stop(): Promise<void> {
    this.keepPlaying = false;
    this.stopStreamOutput();
    await this.pause();
    await this.seek(0);
  }

  private tick = (): void => {
    const t = this.currentTime();
    if (this.playing && !this.hasLoop() && t >= this.duration - 0.02) {
      this.keepPlaying = false;
      this.stopStreamOutput();
      void this.stopSources(false);
      this.offset = this.duration;
    }
    this.onTimeUpdate?.(this.currentTime(), this.duration, this.playing);
    if (this.playing) {
      this.raf = requestAnimationFrame(this.tick);
    }
  };
}

const engine = new StemMixerEngine();
engine.installWakeHooks();
let lastStemKey = "";
let state: MixerState = {
  volumesDb: {},
  pan: {},
  muted: {},
  soloed: {},
  masterVolumeDb: DB_DEFAULT,
};
let stemInfos: StemInfo[] = [];
/** Groups the user folded. Hiding rows never mutes them; this is view state only. */
const collapsedGroups = new Set<string>();
let trackTitle = "";
let metroConfig: MetronomeConfig | null = null;
let metroOptions: MetronomeOptions = coerceMetronomeOptions({
  accent: true,
  rate: 1,
  sound: "classic",
  follow: "smart",
});

let wantPlaying = false;
let transportBusy = false;
let transportPending: "play" | "pause" | "stop" | null = null;

let reportTimer: number | null = null;
let lastPublished = "";
let analysisData: AnalysisData | null = null;
let tabData: TabData | null = null;
type PracticePanelTab = "chords" | "sections" | "lyrics" | "tab";
let practicePanelTab: PracticePanelTab = "chords";
let lastPanelUpdateMs = 0;
let countInBars = 0;
/** Last parent-supplied loop/count-in, so a debounced save cannot undo local edits. */
let lastPracticeLoopArg: string | null = null;
let lastPracticeCountInArg: string | null = null;

type PlaybackPresetId = "all" | "karaoke" | "acapella" | "drumsBass" | "custom";
let activePreset: PlaybackPresetId = "all";

const VIDEO_DRIFT_SEC = 0.4;
const VIDEO_RESYNC_COOLDOWN_MS = 2000;
const HAVE_METADATA = 1;

function videoNeedsSeek(
  audioSec: number,
  videoSec: number,
  nowMs: number,
  lastSeekMs: number
): boolean {
  if (nowMs - lastSeekMs < VIDEO_RESYNC_COOLDOWN_MS) return false;
  return Math.abs(videoSec - audioSec) > VIDEO_DRIFT_SEC;
}

/** Picture timeline position for a stem playhead. Offset is the region start. */
function pictureTimeForMixer(mixerSec: number, offsetSec: number): number {
  const offset = Number.isFinite(offsetSec) && offsetSec > 0 ? offsetSec : 0;
  const mixer = Number.isFinite(mixerSec) && mixerSec > 0 ? mixerSec : 0;
  return mixer + offset;
}

/** Stem playhead for a picture time, clamped to the stem duration. */
function mixerTimeForPicture(
  pictureSec: number,
  offsetSec: number,
  durationSec: number
): number {
  const offset = Number.isFinite(offsetSec) && offsetSec > 0 ? offsetSec : 0;
  const picture = Number.isFinite(pictureSec) ? pictureSec : offset;
  const raw = picture - offset;
  const dur = Number.isFinite(durationSec) && durationSec > 0 ? durationSec : 0;
  if (!(dur > 0)) return Math.max(0, raw);
  return Math.max(0, Math.min(dur, raw));
}

const PICTURE_STABLE_SEC = 0.05;

type PictureSeekKind = "seek-stems" | "pull-picture" | "echo" | "defer" | "hold";

type PictureSeekInput = {
  playing: boolean;
  syncOrigin: "audio" | "video" | null;
  nowMs: number;
  suppressUntilMs: number;
  videoSyncAtMs: number;
  pictureSec: number;
  mixerSec: number;
  offsetSec: number;
  lastPictureSec: number;
  lastMixerSec: number;
  commandedPictureSec: number | null;
  deferredPictureSec: number | null;
  fromSeekedEvent: boolean;
};

/** Who moves when the picture time and the stem playhead disagree. */
function pictureSeekDecision(input: PictureSeekInput): PictureSeekKind {
  const echoWindow = input.nowMs - input.videoSyncAtMs < VIDEO_RESYNC_COOLDOWN_MS;
  const nearCommand =
    input.commandedPictureSec != null &&
    Math.abs(input.pictureSec - input.commandedPictureSec) <= VIDEO_DRIFT_SEC;
  if (input.syncOrigin === "audio" || input.nowMs < input.suppressUntilMs) {
    return "echo";
  }
  if (echoWindow && nearCommand) return "echo";

  let userJump = input.fromSeekedEvent;
  if (!userJump && input.lastPictureSec >= 0 && input.lastMixerSec >= 0) {
    const pictureMoved = input.pictureSec - input.lastPictureSec;
    const mixerMoved = input.mixerSec - input.lastMixerSec;
    userJump = Math.abs(pictureMoved - mixerMoved) > VIDEO_DRIFT_SEC;
  }
  if (userJump) {
    if (input.fromSeekedEvent) return "seek-stems";
    if (echoWindow && !nearCommand) {
      const expected = pictureTimeForMixer(input.mixerSec, input.offsetSec);
      const awayFromMixer = Math.abs(input.pictureSec - expected) > VIDEO_DRIFT_SEC;
      const awayFromCommand =
        input.commandedPictureSec == null ||
        Math.abs(input.pictureSec - input.commandedPictureSec) > VIDEO_DRIFT_SEC;
      const stable =
        input.deferredPictureSec != null &&
        Math.abs(input.pictureSec - input.deferredPictureSec) <= PICTURE_STABLE_SEC;
      if (stable && awayFromMixer && awayFromCommand) return "seek-stems";
      return "defer";
    }
    return "seek-stems";
  }
  if (
    input.playing &&
    input.syncOrigin !== "video" &&
    videoNeedsSeek(
      pictureTimeForMixer(input.mixerSec, input.offsetSec),
      input.pictureSec,
      input.nowMs,
      input.videoSyncAtMs
    )
  ) {
    return "pull-picture";
  }
  return "hold";
}

const VIDEO_MIN_WIDTH = 160;
const VIDEO_STEM_GUTTER = 160;
const VIDEO_LAYOUT_KEY = "audiotools_stem_mixer_video_layout";

type VideoSnap = "left" | "right" | "above" | "below";

type VideoLayout = {
  placed: boolean;
  snap: VideoSnap;
  width: number;
  x: number;
  y: number;
};

function videoSnap(value: unknown): VideoSnap {
  if (value === "left" || value === "right" || value === "below") return value;
  return "above";
}

function dockedVideoLayout(layout: VideoLayout): VideoLayout {
  return {
    placed: false,
    snap: videoSnap(layout.snap),
    width: layout.width,
    x: layout.x,
    y: layout.y,
  };
}

let videoLayout: VideoLayout = {
  placed: false,
  snap: "above",
  width: 0,
  x: 0,
  y: 0,
};

function videoWidthGutter(snap: VideoSnap): number {
  // Stacked Top/Bottom can use nearly the full frame; side-by-side keeps stem room.
  return snap === "above" || snap === "below" ? 8 : VIDEO_STEM_GUTTER;
}

function clampVideoWidth(
  width: number,
  frameWidth: number,
  snap: VideoSnap = videoLayout.snap
): number {
  const gutter = videoWidthGutter(videoSnap(snap));
  const max = Math.max(1, frameWidth - gutter);
  const min = Math.min(VIDEO_MIN_WIDTH, max);
  const w = Number.isFinite(width) ? width : min;
  return Math.max(min, Math.min(max, w));
}

function clampVideoBox(
  x: number,
  y: number,
  width: number,
  frameWidth: number,
  frameHeight: number,
  boxHeight: number,
  snap: VideoSnap = videoLayout.snap
): { x: number; y: number; width: number } {
  const w = clampVideoWidth(width, frameWidth, snap);
  const h =
    Number.isFinite(boxHeight) && boxHeight > 0 ? boxHeight : (w * 9) / 16;
  const maxX = Math.max(0, frameWidth - w);
  const maxY = Math.max(0, frameHeight - h);
  const nx = Number.isFinite(x) ? x : 0;
  const ny = Number.isFinite(y) ? y : 0;
  return {
    width: w,
    x: Math.max(0, Math.min(maxX, nx)),
    y: Math.max(0, Math.min(maxY, ny)),
  };
}

function loadVideoLayout(): void {
  try {
    const raw = sessionStorage.getItem(VIDEO_LAYOUT_KEY);
    if (!raw) return;
    const data = JSON.parse(raw) as Partial<VideoLayout>;
    videoLayout = {
      placed: Boolean(data.placed),
      snap: videoSnap(data.snap),
      width: Number(data.width) > 0 ? Number(data.width) : 0,
      x: Number(data.x) || 0,
      y: Number(data.y) || 0,
    };
  } catch {
    /* private mode */
  }
}

function saveVideoLayout(): void {
  try {
    sessionStorage.setItem(VIDEO_LAYOUT_KEY, JSON.stringify(videoLayout));
  } catch {
    /* quota / private mode */
  }
}

loadVideoLayout();

const STEM_WAVE_DEFAULT = 52;
const STEM_WAVE_MIN = 32;
const STEM_WAVE_MAX = 120;
const STEM_WAVE_KEY = "audiotools_stem_mixer_stem_wave";

let stemWaveHeight = STEM_WAVE_DEFAULT;

function clampStemWaveHeight(height: number): number {
  const h = Number.isFinite(height) ? height : STEM_WAVE_DEFAULT;
  return Math.max(STEM_WAVE_MIN, Math.min(STEM_WAVE_MAX, h));
}

function loadStemWaveHeight(): void {
  try {
    const raw = sessionStorage.getItem(STEM_WAVE_KEY);
    if (!raw) return;
    const data = JSON.parse(raw) as { height?: number };
    if (Number(data.height) > 0) {
      stemWaveHeight = clampStemWaveHeight(Number(data.height));
    }
  } catch {
    /* private mode */
  }
}

function saveStemWaveHeight(): void {
  try {
    sessionStorage.setItem(
      STEM_WAVE_KEY,
      JSON.stringify({ height: stemWaveHeight })
    );
  } catch {
    /* quota / private mode */
  }
}

loadStemWaveHeight();

function pictureIsVisible(): boolean {
  return Boolean(youtubeVideoId || localVideoUrl) && !hideYoutubeVideo;
}

function applyStemWaveHeight(): void {
  const mixer = mixerFrame();
  const handle = document.getElementById("btn-resize-stems");
  if (mixer) {
    mixer.style.setProperty("--stem-wave-height", `${stemWaveHeight}px`);
  }
  if (handle) handle.hidden = !pictureIsVisible();
}

function presetSoloMap(
  ids: string[],
  preset: Exclude<PlaybackPresetId, "custom">
): Record<string, boolean> | null {
  if ((preset === "karaoke" || preset === "acapella") && !ids.includes("vocals")) {
    return null;
  }
  const soloed: Record<string, boolean> = {};
  for (const id of ids) soloed[id] = false;
  if (preset === "all") return soloed;
  if (preset === "acapella") {
    soloed.vocals = true;
    return soloed;
  }
  if (preset === "karaoke") {
    for (const id of ids) {
      if (id !== "vocals" && id !== "metronome") soloed[id] = true;
    }
    return soloed;
  }
  if (ids.includes("drums")) soloed.drums = true;
  if (ids.includes("bass")) soloed.bass = true;
  return soloed;
}

type YTPlayer = {
  playVideo: () => void;
  pauseVideo: () => void;
  seekTo: (seconds: number, allowSeekAhead: boolean) => void;
  getCurrentTime: () => number;
  mute: () => void;
  destroy: () => void;
  setPlaybackRate?: (rate: number) => void;
  getAvailablePlaybackRates?: () => number[];
  getPlaybackRate?: () => number;
};

let youtubeVideoId = "";
let youtubeFallbackId = "";
let localVideoUrl = "";
let localVideoEl: HTMLVideoElement | null = null;
let hideYoutubeVideo = false;
let ytPlayer: YTPlayer | null = null;
let ytMountedId = "";
let pictureMountGen = 0;
let videoSyncAt = 0;
let videoOffsetSec = 0;
let syncOrigin: "audio" | "video" | null = null;
let suppressPictureEventsUntil = 0;
let lastPictureObserved = -1;
let lastMixerObserved = -1;
let commandedPictureSec: number | null = null;
let deferredPictureSec: number | null = null;
let pictureSeekGen = 0;
let picturePollTimer: number | null = null;
let localVideoApplyToken = 0;
let localPlayFromUs = false;
let youtubePlayFromUsUntil = 0;
let localVideoFailedUrl = "";

function ensureYoutubeApi(): Promise<void> {
  const yt = (window as unknown as { YT?: { Player?: unknown } }).YT;
  if (yt?.Player) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const w = window as unknown as { onYouTubeIframeAPIReady?: () => void };
    const prev = w.onYouTubeIframeAPIReady;
    w.onYouTubeIframeAPIReady = () => {
      prev?.();
      resolve();
    };
    if (!document.querySelector("script[data-youtube-iframe]")) {
      const script = document.createElement("script");
      script.src = "https://www.youtube.com/iframe_api";
      script.dataset.youtubeIframe = "1";
      script.onerror = () => reject(new Error("YouTube player failed to load"));
      document.head.appendChild(script);
    }
  });
}

function stopPicturePoll(): void {
  if (picturePollTimer !== null) {
    window.clearInterval(picturePollTimer);
    picturePollTimer = null;
  }
}

function startPicturePoll(): void {
  if (picturePollTimer !== null) return;
  picturePollTimer = window.setInterval(() => {
    if (engine.isPlaying()) return;
    syncPictureToStems(false);
  }, 250);
}

function destroyYoutubePlayer(): void {
  pictureMountGen += 1;
  stopPicturePoll();
  try {
    ytPlayer?.destroy();
  } catch {
    /* frame already gone; stems keep playing */
  }
  ytPlayer = null;
  ytMountedId = "";
}

function destroyLocalVideo(): void {
  localVideoApplyToken += 1;
  if (!localVideoEl) return;
  try {
    localVideoEl.pause();
    localVideoEl.removeAttribute("src");
    localVideoEl.load();
  } catch {
    /* element already gone */
  }
  localVideoEl = null;
}

function showYoutubeFallback(): void {
  youtubeFallbackId = youtubeVideoId;
  destroyYoutubePlayer();
  const slot = document.getElementById("yt-player-slot");
  if (!slot || !youtubeVideoId) return;
  const id = youtubeVideoId;
  slot.replaceChildren();
  const box = document.createElement("div");
  box.className = "mixer-video-fallback";
  const img = document.createElement("img");
  img.alt = "YouTube thumbnail";
  img.src = `https://i.ytimg.com/vi/${encodeURIComponent(id)}/hqdefault.jpg`;
  const link = document.createElement("a");
  link.href = `https://www.youtube.com/watch?v=${encodeURIComponent(id)}`;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = "Open on YouTube";
  box.append(img, link);
  slot.append(box);
}

function markPictureCommand(): void {
  syncOrigin = "audio";
  videoSyncAt = performance.now();
  suppressPictureEventsUntil = videoSyncAt + 400;
  window.setTimeout(() => {
    if (syncOrigin === "audio") syncOrigin = null;
  }, 400);
}

function readPictureTime(): number | null {
  try {
    if (localVideoEl && localVideoUrl && !hideYoutubeVideo) {
      return localVideoEl.currentTime;
    }
    if (ytPlayer && youtubeVideoId && !hideYoutubeVideo) return ytPlayer.getCurrentTime();
  } catch {
    return null;
  }
  return null;
}

/** Closest discrete rate YouTube allows (often 0.25–2). */
function nearestYoutubePlaybackRate(
  desired: number,
  available: number[] | null | undefined
): number {
  const list = (available || []).filter((r) => Number.isFinite(r) && r > 0);
  if (!list.length) return 1;
  let best = list[0]!;
  let bestDist = Math.abs(best - desired);
  for (let i = 1; i < list.length; i++) {
    const r = list[i]!;
    const d = Math.abs(r - desired);
    if (d < bestDist) {
      best = r;
      bestDist = d;
    }
  }
  return best;
}

/**
 * Mirror practice Speed onto the picture so tab/tutorial videos stay locked.
 * Key (pitch) does not change video rate. Audio remains the clock.
 */
function applyPicturePlaybackRate(): void {
  const rate = engine.getEffectivePlaybackRate();
  if (localVideoEl && !hideYoutubeVideo && localVideoUrl) {
    try {
      localVideoEl.playbackRate = rate;
    } catch {
      /* some engines reject extreme rates */
    }
  }
  if (ytPlayer && !hideYoutubeVideo && youtubeVideoId) {
    try {
      const available = ytPlayer.getAvailablePlaybackRates?.() ?? null;
      const ytRate = nearestYoutubePlaybackRate(rate, available);
      ytPlayer.setPlaybackRate?.(ytRate);
    } catch {
      /* keep position sync; never crash the mixer */
    }
  }
}

function writePictureTo(pictureT: number, playing: boolean): void {
  commandedPictureSec = pictureT;
  lastPictureObserved = pictureT;
  lastMixerObserved = engine.currentTime();
  applyPicturePlaybackRate();
  if (localVideoEl && !hideYoutubeVideo && localVideoUrl) {
    const video = localVideoEl;
    video.muted = true;
    if (video.readyState < HAVE_METADATA) {
      const token = ++localVideoApplyToken;
      const onMeta = () => {
        video.removeEventListener("loadedmetadata", onMeta);
        if (token !== localVideoApplyToken || localVideoEl !== video) return;
        markPictureCommand();
        writePictureTo(
          pictureTimeForMixer(engine.currentTime(), videoOffsetSec),
          engine.isPlaying() || wantPlaying
        );
      };
      video.addEventListener("loadedmetadata", onMeta);
      return;
    }
    try {
      if (Math.abs(video.currentTime - pictureT) > VIDEO_DRIFT_SEC) {
        video.currentTime = pictureT;
      }
      if (playing) playLocalVideo(video);
      else video.pause();
    } catch {
      /* picture failed; audio remains the clock */
    }
    return;
  }
  if (ytPlayer && !hideYoutubeVideo && youtubeVideoId) {
    try {
      ytPlayer.mute();
      ytPlayer.seekTo(pictureT, true);
      if (playing) playYoutubeVideo();
      else ytPlayer.pauseVideo();
    } catch {
      /* picture failed; audio remains the clock */
    }
  }
}

function nudgeYoutubeToAudio(): void {
  markPictureCommand();
  writePictureTo(
    pictureTimeForMixer(engine.currentTime(), videoOffsetSec),
    engine.isPlaying()
  );
}

function pauseStemsFromPicture(): void {
  if (!engine.isPlaying()) return;
  if (syncOrigin === "audio") return;
  if (performance.now() < suppressPictureEventsUntil) return;
  wantPlaying = false;
  setPlayPauseLabel(false);
  transportPending = "pause";
  applyTransport();
}

function startStemsFromPicture(): void {
  if (engine.isPlaying() || wantPlaying) return;
  wantPlaying = true;
  setPlayPauseLabel(true);
  transportPending = "play";
  applyTransport();
}

function playLocalVideo(video: HTMLVideoElement): void {
  localPlayFromUs = true;
  queueMicrotask(() => {
    localPlayFromUs = false;
  });
  try {
    const started = video.play();
    if (started) {
      void started.catch(() => {
        /* rejected; the next Play can try again */
      });
    }
  } catch {
    localPlayFromUs = false;
  }
}

function playYoutubeVideo(): void {
  if (!ytPlayer) return;
  youtubePlayFromUsUntil = performance.now() + 400;
  ytPlayer.playVideo();
}

function seekStemsFromPicture(pictureSec: number): void {
  const gen = ++pictureSeekGen;
  syncOrigin = "video";
  videoSyncAt = performance.now();
  deferredPictureSec = null;
  const next = mixerTimeForPicture(pictureSec, videoOffsetSec, engine.getDuration());
  lastPictureObserved = pictureSec;
  lastMixerObserved = next;
  void engine.seek(next).finally(() => {
    if (gen !== pictureSeekGen) return;
    if (syncOrigin === "video") syncOrigin = null;
  });
}

function applyPictureSeek(picture: number, playing: boolean, fromSeekedEvent: boolean): void {
  if (!fromSeekedEvent && syncOrigin === "video") return;
  const mixerT = engine.currentTime();
  const decision = pictureSeekDecision({
    playing,
    syncOrigin,
    nowMs: performance.now(),
    suppressUntilMs: suppressPictureEventsUntil,
    videoSyncAtMs: videoSyncAt,
    pictureSec: picture,
    mixerSec: mixerT,
    offsetSec: videoOffsetSec,
    lastPictureSec: lastPictureObserved,
    lastMixerSec: lastMixerObserved,
    commandedPictureSec,
    deferredPictureSec,
    fromSeekedEvent,
  });
  if (decision === "defer") {
    deferredPictureSec = picture;
    return;
  }
  if (decision === "seek-stems") {
    seekStemsFromPicture(picture);
    return;
  }
  if (decision === "pull-picture") {
    deferredPictureSec = null;
    markPictureCommand();
    writePictureTo(pictureTimeForMixer(mixerT, videoOffsetSec), true);
    return;
  }
  lastPictureObserved = picture;
  lastMixerObserved = mixerT;
  deferredPictureSec = null;
}

function syncPictureToStems(playing: boolean): void {
  if (hideYoutubeVideo) return;
  const picture = readPictureTime();
  if (picture == null) return;
  applyPictureSeek(picture, playing, false);
}

function abandonUnplayableLocalVideo(): void {
  if (!localVideoUrl || !youtubeVideoId) return;
  localVideoFailedUrl = localVideoUrl;
  localVideoUrl = "";
  destroyLocalVideo();
  mountYoutubePlayer();
}

function mountLocalVideo(): void {
  const slot = document.getElementById("yt-player-slot");
  if (!slot || !localVideoUrl || hideYoutubeVideo) {
    destroyLocalVideo();
    return;
  }
  if (localVideoEl && localVideoEl.dataset.src === localVideoUrl) return;
  destroyYoutubePlayer();
  destroyLocalVideo();
  slot.replaceChildren();
  const video = document.createElement("video");
  video.className = "mixer-local-video";
  video.controls = true;
  video.muted = true;
  video.playsInline = true;
  video.preload = "metadata";
  video.addEventListener("error", () => {
    if (localVideoEl !== video) return;
    abandonUnplayableLocalVideo();
  });
  slot.append(video);
  localVideoEl = video;
  video.src = localVideoUrl;
  video.dataset.src = localVideoUrl;
  video.addEventListener("play", () => {
    video.muted = true;
    if (localPlayFromUs) return;
    startStemsFromPicture();
  });
  video.addEventListener("pause", () => {
    pauseStemsFromPicture();
  });
  video.addEventListener("seeked", () => {
    applyPictureSeek(video.currentTime, engine.isPlaying(), true);
  });
  nudgeYoutubeToAudio();
}

function mountYoutubePlayer(): void {
  const slot = document.getElementById("yt-player-slot");
  if (!slot || !youtubeVideoId || hideYoutubeVideo) {
    destroyYoutubePlayer();
    return;
  }
  if (ytPlayer && ytMountedId === youtubeVideoId) return;
  destroyYoutubePlayer();
  slot.replaceChildren();
  const holder = document.createElement("div");
  holder.id = "yt-player";
  slot.appendChild(holder);
  const wanted = youtubeVideoId;
  const gen = ++pictureMountGen;
  void ensureYoutubeApi()
    .then(() => {
      if (gen !== pictureMountGen) return;
      if (hideYoutubeVideo || youtubeVideoId !== wanted) return;
      const YT = (window as unknown as {
        YT?: {
          Player: new (
            el: string,
            opts: Record<string, unknown>
          ) => YTPlayer;
        };
      }).YT;
      if (!YT?.Player || !document.getElementById("yt-player")) return;
      startPicturePoll();
      ytPlayer = new YT.Player("yt-player", {
        videoId: wanted,
        width: 1920,
        height: 1080,
        playerVars: {
          autoplay: 0,
          mute: 1,
          controls: 1,
          disablekb: 1,
          fs: 1,
          modestbranding: 1,
          rel: 0,
          playsinline: 1,
          vq: "hd1080",
        },
        events: {
          onReady: (ev: { target: YTPlayer }) => {
            try {
              ev.target.mute();
              ev.target.pauseVideo();
            } catch {
              /* audio still plays */
            }
            applyPicturePlaybackRate();
            nudgeYoutubeToAudio();
          },
          onStateChange: (ev: { data: number }) => {
            if (ev.data === 1) {
              try {
                ytPlayer?.mute();
              } catch {
                /* audio remains the clock */
              }
              if (performance.now() < youtubePlayFromUsUntil) return;
              startStemsFromPicture();
            }
            if (ev.data === 2) pauseStemsFromPicture();
          },
          onError: () => {
            showYoutubeFallback();
          },
        },
      });
      ytMountedId = wanted;
    })
    .catch(() => {
      showYoutubeFallback();
    });
}

function updateYoutubeChrome(): void {
  const wrap = document.getElementById("mixer-video-wrap");
  const showBtn = document.getElementById("btn-show-video");
  const hasVideo = Boolean(youtubeVideoId || localVideoUrl);
  if (wrap) wrap.hidden = !hasVideo || hideYoutubeVideo;
  if (showBtn) showBtn.hidden = !hasVideo || !hideYoutubeVideo;
  applyStemWaveHeight();
}

function mountPicture(): void {
  if (localVideoUrl) mountLocalVideo();
  else if (youtubeVideoId) mountYoutubePlayer();
  else {
    destroyYoutubePlayer();
    destroyLocalVideo();
  }
}

function applyYoutubeArgs(
  id: string,
  hide: boolean,
  videoUrl = "",
  offsetSec = 0
): void {
  const nextId = (id || "").trim();
  const rawUrl = (videoUrl || "").trim();
  if (rawUrl !== localVideoFailedUrl) localVideoFailedUrl = "";
  const nextUrl = rawUrl === localVideoFailedUrl ? "" : rawUrl;
  const nextHide = !(nextId || nextUrl) || hide;
  const nextOffset = Number.isFinite(offsetSec) && offsetSec > 0 ? offsetSec : 0;
  if (nextId !== youtubeVideoId || nextHide) youtubeFallbackId = "";
  const changed =
    nextId !== youtubeVideoId ||
    nextUrl !== localVideoUrl ||
    nextHide !== hideYoutubeVideo;
  const offsetChanged = nextOffset !== videoOffsetSec;
  youtubeVideoId = nextId;
  localVideoUrl = nextUrl;
  hideYoutubeVideo = nextHide;
  videoOffsetSec = nextOffset;
  updateYoutubeChrome();
  if (
    !changed &&
    !offsetChanged &&
    (ytPlayer || localVideoEl || (youtubeFallbackId && youtubeFallbackId === nextId))
  ) {
    return;
  }
  if (nextHide || (!nextId && !nextUrl)) {
    destroyYoutubePlayer();
    destroyLocalVideo();
    return;
  }
  if (!changed && offsetChanged && (ytPlayer || localVideoEl)) {
    nudgeYoutubeToAudio();
    return;
  }
  mountPicture();
}

const SLEEP_RESUME_HINT = "Tap Play to resume after sleep";

engine.setSoftPauseCallback(() => {
  /* Blur and sleep are not a user Pause — keepPlaying recovery handles them. */
});

function mediaSessionArtwork(): MediaImage[] {
  const id = (youtubeVideoId || "").trim();
  if (!id) return [];
  return [
    {
      src: `https://i.ytimg.com/vi/${encodeURIComponent(id)}/hqdefault.jpg`,
      sizes: "480x360",
      type: "image/jpeg",
    },
  ];
}

function clearMediaSession(): void {
  stopKeepAliveAudio();
  const ms = navigator.mediaSession;
  if (!ms) return;
  try {
    ms.metadata = null;
    ms.playbackState = "none";
    for (const action of [
      "play",
      "pause",
      "seekto",
      "seekbackward",
      "seekforward",
    ] as MediaSessionAction[]) {
      try {
        ms.setActionHandler(action, null);
      } catch {
        /* unsupported action */
      }
    }
  } catch {
    /* Media Session unavailable */
  }
}

function bindMediaSessionHandlers(): void {
  const ms = navigator.mediaSession;
  if (!ms) return;
  try {
    ms.setActionHandler("play", () => {
      wantPlaying = true;
      setPlayPauseLabel(true);
      transportPending = "play";
      applyTransport();
    });
    ms.setActionHandler("pause", () => {
      wantPlaying = false;
      setPlayPauseLabel(false);
      transportPending = "pause";
      applyTransport();
    });
    ms.setActionHandler("seekto", (details) => {
      if (details.seekTime == null || !Number.isFinite(details.seekTime)) return;
      syncOrigin = "audio";
      void engine.seek(details.seekTime).then(() => nudgeYoutubeToAudio());
    });
    ms.setActionHandler("seekbackward", (details) => {
      const delta = details.seekOffset ?? 10;
      syncOrigin = "audio";
      void engine
        .seek(Math.max(0, engine.currentTime() - delta))
        .then(() => nudgeYoutubeToAudio());
    });
    ms.setActionHandler("seekforward", (details) => {
      const delta = details.seekOffset ?? 10;
      syncOrigin = "audio";
      void engine
        .seek(engine.currentTime() + delta)
        .then(() => nudgeYoutubeToAudio());
    });
  } catch {
    /* Media Session actions unavailable */
  }
}

function publishMediaSession(playing: boolean): void {
  const ms = navigator.mediaSession;
  if (!ms) return;
  try {
    bindMediaSessionHandlers();
    ms.metadata = new MediaMetadata({
      title: trackTitle.trim() || "Stem mix",
      artist: "Audio Tools",
      artwork: mediaSessionArtwork(),
    });
    ms.playbackState = playing ? "playing" : "paused";
  } catch {
    /* Media Session unavailable */
  }
}

function updateMediaSessionPosition(): void {
  const ms = navigator.mediaSession;
  if (!ms?.setPositionState) return;
  const dur = engine.getDuration();
  if (!(dur > 0)) return;
  try {
    ms.setPositionState({
      duration: dur,
      playbackRate: engine.isPracticeAvailable()
        ? engine.getPracticeSettings().rate
        : 1,
      position: Math.min(dur, Math.max(0, engine.currentTime())),
    });
  } catch {
    /* position not accepted yet */
  }
}

const TRANSPORT_STORAGE_KEY = "audiotools_stem_mixer_transport";

function saveTransport(): void {
  try {
    sessionStorage.setItem(
      TRANSPORT_STORAGE_KEY,
      JSON.stringify({
        stemKey: lastStemKey,
        offset: engine.currentTime(),
        playing: engine.isPlaying(),
      })
    );
  } catch {
    /* ignore quota / private mode */
  }
}

function restoreTransport(): void {
  try {
    const raw = sessionStorage.getItem(TRANSPORT_STORAGE_KEY);
    if (!raw) return;
    const data = JSON.parse(raw) as {
      stemKey?: string;
      offset?: number;
      playing?: boolean;
    };
    if (data.stemKey !== lastStemKey) return;
    const offset = Number(data.offset) || 0;
    if (offset > 0) {
      void engine.seek(offset);
    }
  } catch {
    /* ignore */
  }
}

function statePayload(): string {
  const practice = engine.getPracticeSettings();
  return JSON.stringify({
    volumesDb: state.volumesDb,
    pan: state.pan,
    muted: state.muted,
    soloed: state.soloed,
    masterVolumeDb: state.masterVolumeDb,
    playbackRate: practice.rate,
    playbackSemitones: practice.semitones,
    loop: engine.getLoopRegion(),
    countInBars: engine.getCountInBars(),
    metronomeOptions: metroConfig ? metroOptions : null,
    hideYoutubeVideo: hideYoutubeVideo,
    videoLayout,
    stemWaveHeight,
  });
}

function scheduleFrameHeight(): void {
  Streamlit.setFrameHeight();
  window.requestAnimationFrame(() => {
    Streamlit.setFrameHeight();
  });
}

function reportState(immediate = false): void {
  const publish = () => {
    const payload = statePayload();
    if (payload === lastPublished) return;
    lastPublished = payload;
    const practice = engine.getPracticeSettings();
    const value: Record<string, unknown> = {
      volumesDb: { ...state.volumesDb },
      pan: { ...state.pan },
      muted: { ...state.muted },
      soloed: { ...state.soloed },
      masterVolumeDb: state.masterVolumeDb,
      playbackRate: practice.rate,
      playbackSemitones: practice.semitones,
      loop: engine.getLoopRegion(),
      countInBars: engine.getCountInBars(),
    };
    if (metroConfig) {
      value.metronomeOptions = { ...metroOptions };
    }
    if (youtubeVideoId || localVideoUrl) {
      value.hideYoutubeVideo = hideYoutubeVideo;
    }
    value.videoLayout = { ...videoLayout };
    value.stemWaveHeight = stemWaveHeight;
    Streamlit.setComponentValue(value);
  };
  if (immediate) {
    if (reportTimer !== null) window.clearTimeout(reportTimer);
    reportTimer = null;
    publish();
    return;
  }
  if (reportTimer !== null) window.clearTimeout(reportTimer);
  // Debounce parent Streamlit reruns while dragging / toggling.
  reportTimer = window.setTimeout(publish, 400);
}

function setPlayPauseLabel(playing: boolean): void {
  const playBtn = document.getElementById("btn-playpause");
  if (!playBtn) return;
  playBtn.textContent = playing ? "Pause" : "Play";
  playBtn.setAttribute("aria-pressed", playing ? "true" : "false");
}

function clearSleepResumeHint(): void {
  const el = document.getElementById("status");
  if (el && el.textContent === SLEEP_RESUME_HINT) {
    el.textContent = "";
  }
}

function applyTransport(): void {
  if (transportBusy) return;
  const pending = transportPending;
  transportPending = null;
  if (pending === "stop") {
    wantPlaying = false;
    setPlayPauseLabel(false);
    transportBusy = true;
    void engine
      .stop()
      .finally(() => {
        clearMediaSession();
        transportBusy = false;
        nudgeYoutubeToAudio();
        if (transportPending) applyTransport();
        else saveTransport();
      });
    return;
  }
  if (wantPlaying === engine.isPlaying()) {
    if (wantPlaying) {
      publishMediaSession(true);
      updateMediaSessionPosition();
    } else {
      clearMediaSession();
    }
    saveTransport();
    return;
  }
  transportBusy = true;
  const op = wantPlaying ? engine.play() : engine.pause();
  void op
    .then(() => {
      const playingNow = engine.isPlaying();
      wantPlaying = playingNow;
      setPlayPauseLabel(playingNow);
      if (playingNow) {
        clearSleepResumeHint();
        publishMediaSession(true);
        updateMediaSessionPosition();
      } else {
        clearMediaSession();
      }
    })
    .catch(() => {
      wantPlaying = engine.isPlaying();
      setPlayPauseLabel(wantPlaying);
      const el = document.getElementById("status");
      if (el && wantPlaying === false) {
        el.textContent = "Tap Play again to start audio";
      }
      if (!wantPlaying) clearMediaSession();
    })
    .finally(() => {
      transportBusy = false;
      nudgeYoutubeToAudio();
      if (transportPending) {
        applyTransport();
        return;
      }
      saveTransport();
    });
}

function togglePlayPause(): void {
  wantPlaying = !wantPlaying;
  setPlayPauseLabel(wantPlaying);
  transportPending = wantPlaying ? "play" : "pause";
  applyTransport();
}

function stopPlayback(): void {
  wantPlaying = false;
  setPlayPauseLabel(false);
  transportPending = "stop";
  applyTransport();
}

/** Put one pan slider and its readout on `pan` (whole steps, -100..100). */
function paintPan(el: HTMLInputElement, pan: number): void {
  el.value = String(pan);
  el.setAttribute("aria-valuetext", panSpoken(pan));
  const label = el.parentElement?.querySelector(".pan-val");
  if (label) label.textContent = formatPan(pan);
}

function applyStateGains(): void {
  engine.applyGains(effectiveGains(stemInfos.map((s) => s.id), state));
  engine.applyPans(state.pan);
  paintStemMeters(engine.currentTime(), engine.getDuration(), engine.isPlaying());
}

function formatTime(sec: number): string {
  if (!isFinite(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/** Drive filled-bar CSS (`--range-fill`) for seek / master range inputs. */
function syncRangeFill(el: HTMLInputElement | null): void {
  if (!el) return;
  const min = Number(el.min);
  const max = Number(el.max);
  const value = Number(el.value);
  const span = max - min;
  const pct =
    Number.isFinite(span) && span > 0 && Number.isFinite(value)
      ? Math.max(0, Math.min(100, ((value - min) / span) * 100))
      : 0;
  el.style.setProperty("--range-fill", `${pct}%`);
}

function mixerFrame(): HTMLElement | null {
  return document.querySelector(".mixer");
}

function updateSnapChrome(): void {
  const dock = document.getElementById("btn-dock-video");
  if (dock) dock.hidden = !videoLayout.placed;
  document.querySelectorAll<HTMLButtonElement>(".mixer-video-toolbar [data-snap]").forEach((el) => {
    const on = el.dataset.snap === videoLayout.snap;
    el.classList.toggle("active", on);
    el.setAttribute("aria-pressed", on ? "true" : "false");
  });
}

function syncStackedStemWidth(snap: VideoSnap, placed: boolean): void {
  const stemsWrap = document.querySelector<HTMLElement>(".mixer-stems-wrap");
  const stage = document.querySelector(".mixer-stage");
  if (!stemsWrap) return;
  const stacked = !placed && (snap === "above" || snap === "below");
  const sized = stacked && videoLayout.width > 0;
  if (stage instanceof HTMLElement) {
    stage.classList.toggle("is-video-sized", sized);
  }
  if (sized) {
    stemsWrap.style.width = `${videoLayout.width}px`;
    stemsWrap.style.maxWidth = "100%";
  } else {
    stemsWrap.style.width = "";
    stemsWrap.style.maxWidth = "";
  }
}

function applyVideoLayout(): void {
  const wrap = document.getElementById("mixer-video-wrap");
  const frame = mixerFrame();
  const stage = document.querySelector(".mixer-stage");
  if (!wrap || !frame) return;
  const snap = videoSnap(videoLayout.snap);
  videoLayout.snap = snap;
  wrap.dataset.snap = snap;
  if (stage instanceof HTMLElement) stage.dataset.snap = snap;
  const frameRect = frame.getBoundingClientRect();
  const frameW = frameRect.width || frame.clientWidth;
  const frameH = frameRect.height || frame.clientHeight;
  // Always reclamp when we have a measured frame so window shrink/grow fits.
  if (videoLayout.width > 0 && frameW > 0) {
    videoLayout.width = clampVideoWidth(videoLayout.width, frameW, snap);
  }
  wrap.classList.toggle("is-sized", videoLayout.width > 0);
  wrap.classList.toggle("is-placed", videoLayout.placed);
  if (!videoLayout.placed) {
    wrap.style.left = "";
    wrap.style.top = "";
    wrap.style.width = videoLayout.width > 0 ? `${videoLayout.width}px` : "";
    syncStackedStemWidth(snap, false);
    updateSnapChrome();
    scheduleFrameHeight();
    return;
  }
  wrap.style.width = `${videoLayout.width}px`;
  syncStackedStemWidth(snap, true);
  if (!(frameW > 0 && frameH > 0)) {
    wrap.style.left = `${videoLayout.x}px`;
    wrap.style.top = `${videoLayout.y}px`;
    updateSnapChrome();
    scheduleFrameHeight();
    return;
  }
  const boxH = wrap.offsetHeight || (videoLayout.width * 9) / 16;
  const clamped = clampVideoBox(
    videoLayout.x,
    videoLayout.y,
    videoLayout.width,
    frameW,
    frameH,
    boxH,
    snap
  );
  videoLayout.x = clamped.x;
  videoLayout.y = clamped.y;
  videoLayout.width = clamped.width;
  wrap.style.width = `${clamped.width}px`;
  wrap.style.left = `${clamped.x}px`;
  wrap.style.top = `${clamped.y}px`;
  updateSnapChrome();
  scheduleFrameHeight();
}

let layoutResizeObserver: ResizeObserver | null = null;
let layoutResizeSaveTimer: number | null = null;
let layoutResizeListening = false;

function onMixerFrameResized(): void {
  applyVideoLayout();
  if (layoutResizeSaveTimer !== null) window.clearTimeout(layoutResizeSaveTimer);
  layoutResizeSaveTimer = window.setTimeout(() => {
    layoutResizeSaveTimer = null;
    saveVideoLayout();
  }, 160);
}

function bindLayoutResizeObserver(): void {
  layoutResizeObserver?.disconnect();
  layoutResizeObserver = null;
  const frame = mixerFrame();
  if (frame && typeof ResizeObserver !== "undefined") {
    layoutResizeObserver = new ResizeObserver(() => {
      onMixerFrameResized();
    });
    layoutResizeObserver.observe(frame);
  }
  if (!layoutResizeListening) {
    layoutResizeListening = true;
    window.addEventListener("resize", onMixerFrameResized);
  }
}

function trackVideoPointer(
  event: PointerEvent,
  onMove: (ev: PointerEvent) => void
): void {
  event.preventDefault();
  const move = (ev: PointerEvent) => onMove(ev);
  const up = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", up);
    saveVideoLayout();
    reportState(true);
    scheduleFrameHeight();
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up);
}

function bindVideoFrameGestures(): void {
  const drag = document.getElementById("btn-drag-video");
  const resize = document.getElementById("btn-resize-video");
  drag?.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    const wrap = document.getElementById("mixer-video-wrap");
    const frame = mixerFrame();
    if (!wrap || !frame) return;
    const frameRect = frame.getBoundingClientRect();
    const rect = wrap.getBoundingClientRect();
    videoLayout.placed = true;
    videoLayout.width = rect.width;
    const originX = rect.left - frameRect.left;
    const originY = rect.top - frameRect.top;
    videoLayout.x = originX;
    videoLayout.y = originY;
    applyVideoLayout();
    const startX = event.clientX;
    const startY = event.clientY;
    trackVideoPointer(event, (ev) => {
      videoLayout.x = originX + (ev.clientX - startX);
      videoLayout.y = originY + (ev.clientY - startY);
      applyVideoLayout();
    });
  });
  resize?.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    const wrap = document.getElementById("mixer-video-wrap");
    const frame = mixerFrame();
    if (!wrap || !frame) return;
    const startW = wrap.getBoundingClientRect().width;
    const startX = event.clientX;
    trackVideoPointer(event, (ev) => {
      const frameRect = frame.getBoundingClientRect();
      videoLayout.width = clampVideoWidth(
        startW + (ev.clientX - startX),
        frameRect.width || frame.clientWidth,
        videoLayout.snap
      );
      applyVideoLayout();
    });
  });
}

function bindStemWaveGestures(): void {
  const resize = document.getElementById("btn-resize-stems");
  resize?.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    if (!pictureIsVisible()) return;
    event.preventDefault();
    const startH = stemWaveHeight;
    const startY = event.clientY;
    const move = (ev: PointerEvent) => {
      stemWaveHeight = clampStemWaveHeight(startH + (ev.clientY - startY));
      applyStemWaveHeight();
      scheduleFrameHeight();
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      saveStemWaveHeight();
      reportState(true);
      scheduleFrameHeight();
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  });
}

function readableTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const total = Math.floor(seconds);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function parseAnalysisArg(raw: unknown): AnalysisData | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  return raw as AnalysisData;
}

function parseTabArg(raw: unknown): TabData | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const tab = raw as TabData;
  return Array.isArray(tab.measures) ? tab : null;
}

function availablePracticeTabs(): PracticePanelTab[] {
  const tabs: PracticePanelTab[] = [];
  if (analysisData?.chords?.length) tabs.push("chords");
  if (analysisData?.sections?.length) tabs.push("sections");
  if (analysisData?.lyrics?.length) tabs.push("lyrics");
  if (tabData?.measures?.length) tabs.push("tab");
  return tabs;
}

function activeSectionAt(t: number): Section | null {
  return sectionAt(analysisData?.sections, t);
}

function sidebarChordPayload(t: number): { symbol: string; diagram: string } {
  const chords = analysisData?.chords;
  let chord = chordAt(chords, t);
  if (!chord && chords && chords.length > 0) {
    let best: (typeof chords)[number] | null = null;
    for (const item of chords) {
      const start = Number(item.start);
      if (Number.isFinite(start) && start <= t && (best === null || start >= Number(best.start))) {
        best = item;
      }
    }
    chord = best ?? chords[0];
  }
  if (!chord) return { symbol: "—", diagram: "" };
  const symbol = formatChordSymbol(chord.symbol);
  const raw = chordDiagramSvg(capoDiagramSymbol(chord.symbol, capoFret)).trim();
  const diagram = raw.startsWith("<svg") && raw.endsWith("</svg>") ? raw : "";
  return { symbol, diagram };
}

function postSidebarChord(t: number): void {
  const payload = sidebarChordPayload(t);
  if (!payload.symbol || payload.symbol === "—") return;
  const key = `${payload.symbol}\n${payload.diagram}`;
  if (key === lastSidebarChord) return;
  lastSidebarChord = key;
  const message = {
    source: "att-sidebar-chord",
    symbol: payload.symbol,
    diagram: payload.diagram,
  };
  try {
    localStorage.setItem("att-sidebar-chord", JSON.stringify(message));
  } catch {
    /* Private mode still gets the postMessage below. */
  }
  const target = window.top ?? window.parent;
  try {
    target.postMessage(message, "*");
  } catch {
    /* The top frame is gone. */
  }
}

function seekPractice(t: number): void {
  const target = Math.max(0, Number(t) || 0);
  void engine.seek(target).then(() => {
    nudgeYoutubeToAudio();
    updatePracticePanelAt(engine.currentTime(), true);
  });
}

function updatePracticeHeader(): void {
  const summary = document.getElementById("practice-summary");
  if (!summary) return;
  const bpm = analysisData?.bpm;
  const key = analysisData?.key;
  const sections = analysisData?.sections?.length ?? 0;
  const parts: string[] = [];
  if (bpm) parts.push(`${Math.round(Number(bpm))} BPM`);
  if (key) parts.push(String(key));
  if (sections) parts.push(`${sections} section${sections === 1 ? "" : "s"}`);
  summary.textContent = parts.join(" · ");
}

function chordDiagramSvg(symbol: string): string {
  const shape = chordShape(symbol);
  if (!shape) return `<div class="chord-no-shape">${escapeHtml(formatChordSymbol(symbol))}</div>`;
  const { width, height, left, right, top, bottom, fretLabelX, dotRadius } =
    CHORD_DIAGRAM_LAYOUT;
  const stringGap = (right - left) / 5;
  const fretGap = (bottom - top) / 4;
  const parts: string[] = [];
  parts.push(
    `<svg class="chord-diagram" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(
      formatChordSymbol(symbol)
    )} chord diagram">`
  );
  if (shape.baseFret <= 1) {
    parts.push(
      `<rect x="${left - 2}" y="${top - 3}" width="${right - left + 4}" height="3" rx="1" class="chord-nut" />`
    );
  } else {
    parts.push(
      `<text x="${fretLabelX}" y="${top + fretGap * 0.7}" text-anchor="end" class="chord-fret-label">${shape.baseFret}</text>`
    );
  }
  for (let i = 0; i < 6; i++) {
    const x = left + i * stringGap;
    parts.push(
      `<line x1="${x}" y1="${top}" x2="${x}" y2="${bottom}" class="chord-string" />`
    );
  }
  for (let f = 0; f <= 4; f++) {
    const y = top + f * fretGap;
    parts.push(
      `<line x1="${left}" y1="${y}" x2="${right}" y2="${y}" class="chord-fret" />`
    );
  }
  shape.frets.forEach((fret, i) => {
    const x = left + i * stringGap;
    if (fret === null) {
      parts.push(
        `<text x="${x}" y="${top - 6}" text-anchor="middle" class="chord-mute">×</text>`
      );
    } else if (fret === 0) {
      parts.push(
        `<circle cx="${x}" cy="${top - 6}" r="3" class="chord-open" />`
      );
    } else {
      const relative = fret - shape.baseFret;
      if (relative >= 0 && relative < 4) {
        const y = top + relative * fretGap + fretGap / 2;
        parts.push(
          `<circle cx="${x}" cy="${y}" r="${dotRadius}" class="chord-dot" />`
        );
      }
    }
  });
  parts.push(`</svg>`);
  return parts.join("");
}

let spelling: "sharp" | "flat" = "sharp";
let listenA432 = false;

function renderPracticePanel(): void {
  const body = document.getElementById("practice-body");
  if (!body) return;
  const active = document.activeElement;
  if (
    active instanceof HTMLElement &&
    body.contains(active) &&
    (active.tagName === "INPUT" || active.tagName === "TEXTAREA")
  ) {
    return;
  }
  const tabs = availablePracticeTabs();
  const buttons = document.querySelectorAll<HTMLButtonElement>(
    "[data-practice-tab]"
  );
  buttons.forEach((button) => {
    const tab = button.dataset.practiceTab as PracticePanelTab;
    button.hidden = !tabs.includes(tab);
    const active = tab === practicePanelTab && tabs.includes(tab);
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });
  if (tabs.length === 0) {
    practicePanelTab = "chords";
    body.innerHTML =
      '<p class="practice-empty">No chords, sections, lyrics, or tab in this run.</p>';
    updatePracticeHeader();
    scheduleFrameHeight();
    return;
  }
  if (!tabs.includes(practicePanelTab)) practicePanelTab = tabs[0]!;
  const sections = analysisData?.sections ?? [];
  const chords = analysisData?.chords ?? [];
  const lyrics = analysisData?.lyrics ?? [];
  const measures = tabData?.measures ?? [];

  if (practicePanelTab === "chords") {
    if (chords.length === 0) {
      body.innerHTML =
        '<p class="practice-empty">No chords in this run.</p>';
    } else {
      const chips = chords
        .map(
          (chord, index) =>
            `<div class="chord-chip" data-chord-index="${index}">` +
            `<input class="chord-chip-symbol" data-chord-index="${index}" value="${escapeHtml(spellChordSymbol(displaySymbol(chord, chordLevel), spelling))}" aria-label="Chord symbol" />` +
            `<span class="chord-difficulty">${escapeHtml(chordDifficulty(chord.symbol))}</span>` +
            `<button type="button" class="chord-chip-time" data-seek="${chord.start}">${readableTime(chord.start)}</button>` +
            `</div>`
        )
        .join("");
      body.innerHTML =
        '<div class="chord-now">' +
        '<div class="chord-now-symbol" id="chord-now-symbol">—</div>' +
        '<div class="chord-now-diagram" id="chord-now-diagram"></div>' +
        '<div class="chord-now-meta"><span id="chord-now-section">—</span>' +
        '<label class="capo-label"><span class="visually-hidden">Capo fret</span>' +
        '<select id="chord-capo" aria-label="Capo fret">' +
        '<option value="0">No capo</option>' +
        Array.from({ length: 11 }, (_, index) => `<option value="${index + 1}">Capo ${index + 1}</option>`).join("") +
        "</select></label>" +
        `<div class="capo-presets" role="group" aria-label="Capo presets">` +
        CAPO_PRESETS.map(
          (preset) =>
            `<button type="button" class="ghost practice-small" data-capo-preset="${preset.fret}">${preset.label}</button>`
        ).join("") +
        `</div>` +
        `<button type="button" class="ghost practice-small" id="btn-spelling" aria-pressed="${spelling === "flat" ? "true" : "false"}">` +
        `${spelling === "flat" ? "Flats" : "Sharps"}</button>` +
        `<label class="capo-label"><span class="visually-hidden">Chord detail</span>` +
        `<select id="chord-level" aria-label="Chord detail">` +
        `<option value="simple"${chordLevel === "simple" ? " selected" : ""}>Simple</option>` +
        `<option value="medium"${chordLevel === "medium" ? " selected" : ""}>Medium</option>` +
        `<option value="chart"${chordLevel === "chart" ? " selected" : ""}>Chart</option>` +
        `</select></label>` +
        `<button type="button" class="ghost practice-small" id="chord-follow">Follow</button>` +
        "</div>" +
        "</div>" +
        `<div class="chord-strip" id="chord-strip">${chips}</div>` +
        '<p class="practice-note">Demo — chord chart is a work in progress; roots can drift on dense mixes — verify by ear.</p>';
    }
  } else if (practicePanelTab === "sections") {
    body.innerHTML = sections.length
      ? `<div class="section-list">${sections
          .map(
            (section, index) =>
              `<div class="section-row" data-section-index="${index}">` +
              `<button type="button" class="section-seek" data-seek="${section.start}">` +
              `<span class="section-time">${readableTime(section.start)}</span>` +
              `</button>` +
              `<input class="section-label" data-section-index="${index}" value="${escapeHtml(section.label)}" aria-label="Section name" />` +
              `<button type="button" class="ghost practice-small section-loop" data-loop-index="${index}">Loop</button>` +
              `</div>`
          )
          .join("")}</div>`
      : '<p class="practice-empty">No sections in this run.</p>';
  } else if (practicePanelTab === "lyrics") {
    body.innerHTML = lyrics.length
      ? `<div class="lyrics-list">${lyrics
          .map((line, index) => {
            const separator = /[\u0e00-\u0e7f\u1000-\u109f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/.test(
              line.text
            )
              ? ""
              : " ";
            const words = line.words?.length
              ? line.words
                  .map(
                    (word) =>
                      `<span class="lyric-word" data-word-start="${word.start}" data-word-end="${word.end}">${escapeHtml(
                        word.text
                      )}</span>`
                  )
                  .join(separator)
              : escapeHtml(line.text);
            return `<div class="lyric-line-row"><button type="button" class="lyric-line" data-line-index="${index}" data-seek="${line.start}">${words}</button>` +
              `<input class="lyric-edit" data-lyric-index="${index}" value="${escapeHtml(line.text)}" aria-label="Lyric line" /></div>`;
          })
          .join("")}</div>`
      : '<p class="practice-empty">No lyrics found (instrumental, or the lyrics model is unavailable).</p>';
  } else {
    const bassPitches = (tabData as { bassPitches?: number[] } | null)?.bassPitches;
    const bass = Array.isArray(bassPitches) && bassPitches.length ? formatBassTab(bassPitches) : "";
    body.innerHTML = measures.length
      ? `<div class="tab-measure-wrap"><div class="tab-measure-label" id="tab-measure-label">Measure</div>` +
        `<pre class="tab-measure" id="tab-measure-now"></pre>` +
        measures
          .map(
            (measure, index) =>
              `<label class="tab-edit-label">Edit ${index + 1}` +
              `<input class="tab-edit" data-measure-index="${index}" value="${escapeHtml(measure.text)}" aria-label="Tab measure ${index + 1}" />` +
              `</label>`
          )
          .join("") +
        `<div class="tab-measure-label">Next</div><pre class="tab-measure tab-measure-next" id="tab-measure-next"></pre>` +
        (bass
          ? `<div class="tab-measure-label">Bass</div><pre class="tab-measure">${escapeHtml(bass)}</pre>`
          : Array.isArray(bassPitches)
            ? `<p class="practice-empty">No bass tab in this run.</p>`
            : "") +
        `</div>` +
        (tabData && (tabData as { pdfUrl?: string }).pdfUrl
          ? `<a class="tab-pdf-link" href="${escapeHtml(
              (tabData as { pdfUrl?: string }).pdfUrl!
            )}" target="_blank" rel="noopener">Open tab PDF</a>`
          : "")
      : '<p class="practice-empty">No tab for this run. Re-run with guitar separation.</p>';
  }

  body.querySelectorAll<HTMLElement>("[data-seek]").forEach((el) => {
    el.onclick = () => seekPractice(Number(el.dataset.seek));
  });
  body.querySelectorAll<HTMLInputElement>("input.chord-chip-symbol").forEach((input) => {
    input.onkeydown = (event) => {
      event.stopPropagation();
      if (event.key === "Enter") input.blur();
    };
    input.onclick = (event) => event.stopPropagation();
    input.onchange = () => {
      const index = Number(input.dataset.chordIndex);
      const chords = analysisData?.chords;
      const chord = chords?.[index];
      if (!chord) return;
      const next = replaceChordSymbol(chord.symbol, input.value);
      if (!next) {
        input.value = formatChordSymbol(chord.symbol);
        return;
      }
      chord.symbol = next;
      chord.simple = "";
      chord.medium = "";
      chordSymbolEdits.set(index, next);
      input.value = next;
      const diagramEl = document.getElementById("chord-now-diagram");
      if (diagramEl) delete diagramEl.dataset.symbol;
      updatePracticePanelAt(engine.currentTime(), true);
      postSidebarChord(engine.currentTime());
    };
  });
  const capoSelect = document.getElementById("chord-capo") as HTMLSelectElement | null;
  if (capoSelect) {
    capoSelect.value = String(capoFret);
    capoSelect.onchange = () => {
      capoFret = Math.max(0, Math.min(11, Math.round(Number(capoSelect.value) || 0)));
      const diagramEl = document.getElementById("chord-now-diagram");
      if (diagramEl) delete diagramEl.dataset.symbol;
      updatePracticePanelAt(engine.currentTime(), true);
      postSidebarChord(engine.currentTime());
    };
  }
  body.querySelectorAll<HTMLButtonElement>("[data-capo-preset]").forEach((button) => {
    button.onclick = () => {
      capoFret = Math.max(0, Math.min(11, Number(button.dataset.capoPreset) || 0));
      if (capoSelect) capoSelect.value = String(capoFret);
      const diagramEl = document.getElementById("chord-now-diagram");
      if (diagramEl) delete diagramEl.dataset.symbol;
      updatePracticePanelAt(engine.currentTime(), true);
      postSidebarChord(engine.currentTime());
    };
  });
  const levelSelect = document.getElementById("chord-level") as HTMLSelectElement | null;
  if (levelSelect) {
    levelSelect.value = chordLevel;
    levelSelect.onchange = () => {
      const next = levelSelect.value;
      chordLevel = next === "simple" || next === "medium" ? next : "chart";
      chordActiveChip = null;
      renderPracticePanel();
      updatePracticePanelAt(engine.currentTime(), true);
    };
  }
  const followButton = document.getElementById("chord-follow");
  if (followButton) {
    followButton.onclick = () => {
      chordUserScrollAt = 0;
      highlightActiveChord(engine.currentTime(), true);
    };
  }
  const chordStrip = document.getElementById("chord-strip");
  if (chordStrip) {
    chordStrip.addEventListener("scroll", () => {
      if (chordProgrammaticScroll) return;
      chordUserScrollAt = performance.now();
    });
  }
  const spellingButton = document.getElementById("btn-spelling");
  if (spellingButton) {
    spellingButton.onclick = () => {
      spelling = spelling === "flat" ? "sharp" : "flat";
      renderPracticePanel();
    };
  }
  body.querySelectorAll<HTMLInputElement>("input.section-label").forEach((input) => {
    input.onkeydown = (event) => event.stopPropagation();
    input.onchange = () => {
      const index = Number(input.dataset.sectionIndex);
      const section = analysisData?.sections?.[index];
      if (!section) return;
      section.label = replaceSectionLabel(section.label, input.value);
      input.value = section.label;
    };
  });
  body.querySelectorAll<HTMLInputElement>("input.lyric-edit").forEach((input) => {
    input.onkeydown = (event) => event.stopPropagation();
    input.onchange = () => {
      const index = Number(input.dataset.lyricIndex);
      const line = analysisData?.lyrics?.[index];
      if (!line) return;
      line.text = replaceLyricLine(line.text, input.value);
      input.value = line.text;
    };
  });
  body.querySelectorAll<HTMLInputElement>("input.tab-edit").forEach((input) => {
    input.onkeydown = (event) => event.stopPropagation();
    input.onchange = () => {
      const index = Number(input.dataset.measureIndex);
      const measure = tabData?.measures?.[index];
      if (!measure) return;
      const next = replaceMeasureText(measure.text, input.value);
      if (!next) {
        input.value = measure.text;
        return;
      }
      measure.text = next;
      input.value = next;
    };
  });
  body.querySelectorAll<HTMLElement>("[data-word-start]").forEach((el) => {
    el.onclick = (event) => {
      // The line button also seeks. A word click is more precise.
      event.stopPropagation();
      seekPractice(Number(el.dataset.wordStart));
    };
  });
  body.querySelectorAll<HTMLElement>("[data-loop-index]").forEach((el) => {
    el.onclick = () => {
      const section = sections[Number(el.dataset.loopIndex)];
      if (section) setLoopFromSection(section);
    };
  });
  updatePracticeHeader();
  updatePracticePanelAt(engine.currentTime(), true);
  scheduleFrameHeight();
}

function setPracticeTab(tab: PracticePanelTab): void {
  if (!availablePracticeTabs().includes(tab)) return;
  practicePanelTab = tab;
  renderPracticePanel();
}

function setLoopFromSection(section: Section): void {
  const region = clampLoop(section.start, section.end, engine.getDuration());
  if (!region) {
    const toggle = document.getElementById("loop-toggle") as HTMLInputElement | null;
    if (toggle) toggle.checked = engine.hasLoop();
    showLoopStatus("That section is too short to loop.");
    return;
  }
  engine.setLoopRegion(region.start, region.end);
  const toggle = document.getElementById("loop-toggle") as HTMLInputElement | null;
  if (toggle) toggle.checked = true;
  showLoopStatus();
  reportState(true);
}

function setLoopFromCurrentSection(): void {
  const section = activeSectionAt(engine.currentTime());
  if (section) {
    setLoopFromSection(section);
    return;
  }
  const t = engine.currentTime();
  const end = Math.min(engine.getDuration(), t + 8);
  const region = clampLoop(t, end, engine.getDuration());
  if (region) {
    engine.setLoopRegion(region.start, region.end);
    const toggle = document.getElementById("loop-toggle") as HTMLInputElement | null;
    if (toggle) toggle.checked = true;
    showLoopStatus();
    reportState(true);
  } else {
    const toggle = document.getElementById("loop-toggle") as HTMLInputElement | null;
    if (toggle) toggle.checked = engine.hasLoop();
    showLoopStatus("Play a little first, then set a loop.");
  }
}

function clearLoop(): void {
  engine.clearLoop();
  const toggle = document.getElementById("loop-toggle") as HTMLInputElement | null;
  if (toggle) toggle.checked = false;
  showLoopStatus("Loop off");
  reportState(true);
}

function showLoopStatus(message?: string): void {
  const el = document.getElementById("loop-status");
  if (!el) return;
  const region = engine.getLoopRegion();
  if (message) {
    el.textContent = message;
  } else if (region) {
    el.textContent = `Loop ${readableTime(region.start)}–${readableTime(region.end)}`;
  } else {
    el.textContent = "Loop off";
  }
}

function setCountInBars(bars: number): void {
  countInBars = Math.max(0, Math.min(2, Math.round(Number(bars) || 0)));
  engine.setCountInBars(countInBars);
  const select = document.getElementById("count-in-bars") as HTMLSelectElement | null;
  if (select) select.value = String(countInBars);
  reportState(true);
}

function syncCountInControl(): void {
  const select = document.getElementById("count-in-bars") as HTMLSelectElement | null;
  if (select) select.value = String(countInBars);
}

function highlightActiveChord(audioTime: number, forceFollow = false): void {
  if (practicePanelTab !== "chords") return;
  const chords = analysisData?.chords ?? [];
  if (chords.length === 0) return;
  const chartTime = chartTimeSeconds(audioTime, engine.chartLatencyMs());
  const next = advanceChordIndex(chords, chordFollowIndex, chartTime);
  const strip = document.getElementById("chord-strip");
  const chip = strip?.querySelector<HTMLElement>(`[data-chord-index="${next}"]`) ?? null;
  if (chip !== chordActiveChip) {
    chordActiveChip?.classList.remove("active");
    chip?.classList.add("active");
    chordActiveChip = chip;
  }
  chordFollowIndex = next;
  if (!strip || !chip) return;
  if (!forceFollow && performance.now() - chordUserScrollAt < 3000) return;
  const left = scrollLeftToCenter(strip, chip);
  if (Math.abs(strip.scrollLeft - left) <= 1) return;
  chordProgrammaticScroll = true;
  strip.scrollLeft = left;
  chordProgrammaticScroll = false;
}

function updatePracticePanelAt(t: number, force = false): void {
  highlightActiveChord(t);
  const now = performance.now();
  if (!force && now - lastPanelUpdateMs < 96) return;
  lastPanelUpdateMs = now;

  const countEl = document.getElementById("count-in-status");
  if (countEl) {
    const remaining = engine.countInRemaining();
    if (remaining > 0.02) {
      countEl.hidden = false;
      const bpm = Number(analysisData?.bpm) || 120;
      // The click buffer starts ~30 ms after now; subtract that lead so the
      // first tick reads the bar's beat count, not one extra.
      const beats = Math.ceil(remaining / (60 / bpm) - 0.05);
      countEl.textContent = `Count-in… ${Math.max(1, beats)}`;
    } else {
      countEl.hidden = true;
      countEl.textContent = "";
    }
  }

  if (practicePanelTab === "chords") {
    const chord = chordAt(analysisData?.chords, chartTimeSeconds(t, engine.chartLatencyMs()));
    const symbol = chord ? formatChordSymbol(displaySymbol(chord, chordLevel)) : "—";
    const symbolEl = document.getElementById("chord-now-symbol");
    const diagramEl = document.getElementById("chord-now-diagram");
    const sectionEl = document.getElementById("chord-now-section");
    if (symbolEl) symbolEl.textContent = symbol;
    if (diagramEl && chord) {
      const shapeSymbol = capoDiagramSymbol(displaySymbol(chord, chordLevel), capoFret);
      const drawn = `${shapeSymbol}|${capoFret}`;
      if (diagramEl.dataset.symbol !== drawn) {
        diagramEl.dataset.symbol = drawn;
        diagramEl.innerHTML = chordDiagramSvg(shapeSymbol);
      }
    } else if (diagramEl) {
      diagramEl.innerHTML = "";
      diagramEl.dataset.symbol = "";
    }
    if (sectionEl) {
      sectionEl.textContent = activeSectionAt(t)?.label ?? "—";
    }
  } else if (practicePanelTab === "sections") {
    const current = activeSectionAt(t);
    const currentIndex = current
      ? (analysisData?.sections ?? []).indexOf(current)
      : -1;
    document.querySelectorAll<HTMLElement>(".section-row").forEach((row) => {
      row.classList.toggle(
        "active",
        Number(row.dataset.sectionIndex) === currentIndex
      );
    });
  } else if (practicePanelTab === "lyrics") {
    const line = lyricLineAt(analysisData?.lyrics, t);
    document.querySelectorAll<HTMLElement>(".lyric-line").forEach((el) => {
      el.classList.toggle("active", el === lineElementFor(line));
    });
    document.querySelectorAll<HTMLElement>(".lyric-word").forEach((word) => {
      const start = Number(word.dataset.wordStart);
      const end = Number(word.dataset.wordEnd);
      word.classList.toggle("active", t >= start && t < end);
    });
  } else if (practicePanelTab === "tab") {
    const measure = measureAt(tabData?.measures, t);
    const activeIndex = measure
      ? tabData!.measures!.indexOf(measure)
      : -1;
    const nowEl = document.getElementById("tab-measure-now");
    const nextEl = document.getElementById("tab-measure-next");
    const labelEl = document.getElementById("tab-measure-label");
    if (nowEl) nowEl.textContent = measure?.text ?? "";
    if (nextEl) {
      nextEl.textContent =
        activeIndex >= 0 ? tabData!.measures![activeIndex + 1]?.text ?? "" : "";
    }
    if (labelEl) {
      labelEl.textContent =
        activeIndex >= 0 ? `Measure ${activeIndex + 1}` : "Measure";
    }
  }
}

function lineElementFor(line: unknown): HTMLElement | null {
  if (!line) return null;
  const index = analysisData?.lyrics?.indexOf(line as never) ?? -1;
  if (index < 0) return null;
  return document.querySelector<HTMLElement>(
    `.lyric-line[data-line-index="${index}"]`
  );
}

function bindPracticePanel(): void {
  document.querySelectorAll<HTMLButtonElement>("[data-practice-tab]").forEach((button) => {
    button.onclick = () =>
      setPracticeTab(button.dataset.practiceTab as PracticePanelTab);
  });
  const loopToggle = document.getElementById("loop-toggle") as HTMLInputElement | null;
  if (loopToggle) {
    loopToggle.onchange = () => {
      if (loopToggle.checked) {
        setLoopFromCurrentSection();
      } else {
        clearLoop();
      }
    };
  }
  const loopSection = document.getElementById("btn-loop-section");
  if (loopSection) loopSection.onclick = () => setLoopFromCurrentSection();
  const loopClear = document.getElementById("btn-loop-clear");
  if (loopClear) loopClear.onclick = () => clearLoop();
  const countIn = document.getElementById("count-in-bars") as HTMLSelectElement | null;
  if (countIn) {
    countIn.onchange = () => setCountInBars(Number(countIn.value));
    syncCountInControl();
  }
  showLoopStatus();
}

function applyPracticeArgs(args: {
  analysis?: unknown;
  tab?: unknown;
  initialLoop?: { start?: number; end?: number } | null;
  countInBars?: number;
}): void {
  const nextAnalysis = parseAnalysisArg(args.analysis);
  const nextTab = parseTabArg(args.tab);
  const dataChanged =
    JSON.stringify(nextAnalysis) !== JSON.stringify(analysisData) ||
    JSON.stringify(nextTab) !== JSON.stringify(tabData);
  analysisData = adoptAnalysis(nextAnalysis);
  tabData = nextTab;
  if (nextAnalysis) {
    engine.setCountInGrid(
      nextAnalysis.bpm,
      nextAnalysis.beats_per_measure,
      metroOptions
    );
  }
  // Apply restored practice args only when the parent's value changed. A
  // debounced project save can still carry the previous value on the rerun
  // that follows a local edit; reapplying it would silently undo the edit.
  const countInArg = Number(args.countInBars);
  const countInKey = JSON.stringify(Number.isFinite(countInArg) ? countInArg : null);
  if (countInKey !== lastPracticeCountInArg) {
    lastPracticeCountInArg = countInKey;
    const initialCountIn = Number(args.countInBars);
    if (Number.isFinite(initialCountIn)) {
      countInBars = Math.max(0, Math.min(2, Math.round(initialCountIn)));
    }
    engine.setCountInBars(countInBars);
    syncCountInControl();
  }
  const loop = args.initialLoop;
  const loopKey = JSON.stringify(loop ?? null);
  if (loopKey !== lastPracticeLoopArg && engine.getDuration() > 0) {
    lastPracticeLoopArg = loopKey;
    if (loop && Number.isFinite(Number(loop.start)) && Number.isFinite(Number(loop.end))) {
      const region = clampLoop(Number(loop.start), Number(loop.end), engine.getDuration());
      if (region) engine.setLoopRegion(region.start, region.end);
    } else {
      engine.clearLoop();
    }
  }
  const toggle = document.getElementById("loop-toggle") as HTMLInputElement | null;
  if (toggle) toggle.checked = Boolean(engine.getLoopRegion());
  showLoopStatus();
  if (dataChanged) {
    lastSidebarChord = "";
    renderPracticePanel();
  }
  postSidebarChord(engine.currentTime());
  if (tabData && (tabData as { pdfUrl?: string }).pdfUrl) {
    // The PDF link is rendered by renderPracticePanel; no extra wiring needed.
  }
}

async function applyInitialPractice(args: {
  initialRate?: number;
  initialSemitones?: number;
}): Promise<void> {
  const current = engine.getPracticeSettings();
  const rate = clampPracticeRate(Number(args.initialRate ?? current.rate));
  const semitones = clampPracticeSemitones(
    Number(args.initialSemitones ?? current.semitones)
  );
  if (rate !== current.rate) {
    const result = await engine.setPracticeRate(rate);
    if (!result.ok && result.message) showPracticeStatus(result.message);
  }
  if (semitones !== engine.getPracticeSettings().semitones) {
    const result = await engine.setPracticeSemitones(semitones);
    if (!result.ok && result.message) showPracticeStatus(result.message);
  }
  const settings = engine.getPracticeSettings();
  syncPracticeChrome(settings);
  savePracticeSettings(settings);
}

function renderUI(theme?: Theme): void {
  const root = document.getElementById("root");
  if (!root) return;
  destroyYoutubePlayer();
  destroyLocalVideo();

  const textColor = theme?.textColor ?? "#31333F";
  const bg = theme?.backgroundColor ?? "#ffffff";
  const secondary = theme?.secondaryBackgroundColor ?? "#f0f2f6";
  const primary = theme?.primaryColor ?? "#ff4b4b";

  root.innerHTML = `
    <div class="mixer" data-ui="v2" tabindex="-1" style="--text:${textColor};--bg:${bg};--secondary:${secondary};--primary:${primary};--waveform-fill:${primary}">
      <button type="button" id="mixer-menu-backdrop" aria-label="Close menu" hidden></button>
      <header class="mixer-header">
        <div class="track-title" id="track-title"></div>
        <p class="mixer-status" id="status" aria-live="polite">Ready</p>
        <div class="mixer-load" id="mixer-load" hidden>
          <div class="mixer-load-bar" id="mixer-load-bar"></div>
        </div>
      </header>
      <div class="mixer-transport">
        <div class="mixer-transport-controls">
          <button type="button" class="primary live-play" id="btn-playpause" aria-pressed="false">Play</button>
          <button type="button" class="ghost" id="btn-stop">Stop</button>
          <button type="button" class="ghost" id="btn-restart">Restart</button>
          <span class="mixer-time" id="time" aria-label="Playback time">
            <span id="time-current">0:00</span>
            <span class="mixer-time-sep" aria-hidden="true">/</span>
            <span id="time-duration">0:00</span>
          </span>
        </div>
        <label class="mixer-seek">
          <span class="visually-hidden">Seek</span>
          <input type="range" id="seek" class="slider-seek" min="0" max="1000" value="0" step="1" disabled />
        </label>
        <div class="mixer-menu-anchor">
          <button type="button" class="ghost" id="btn-practice-sheet" aria-expanded="true" aria-controls="practice-panel">Practice</button>
          <button type="button" class="ghost" id="btn-mix" aria-expanded="false" aria-controls="mix-menu">Mix</button>
          <button type="button" class="ghost" id="btn-speed-key" aria-expanded="false" aria-controls="practice-speed-popover">Speed &amp; Key</button>
          <div id="mix-menu" hidden>
            <div class="mixer-presets" role="group" aria-label="Mix presets">
              <button type="button" class="preset-chip" data-preset="all">All</button>
              <button type="button" class="preset-chip" data-preset="karaoke">Karaoke</button>
              <button type="button" class="preset-chip" data-preset="acapella">Acapella</button>
              <button type="button" class="preset-chip" data-preset="drumsBass">Drums + Bass</button>
            </div>
            <button type="button" class="ghost" id="btn-muteall">Mute all</button>
            <button type="button" class="ghost" id="btn-reset">Reset mix</button>
            <button type="button" class="ghost" id="btn-clearsolo" hidden>Clear solo</button>
            <button type="button" class="ghost" id="btn-show-video" hidden>Show video</button>
          </div>
          <div class="mixer-practice" id="practice-speed-popover" role="group" aria-label="Speed and Key" hidden>
            <div class="practice-heading">Speed &amp; Key</div>
            <div class="practice-row">
              <span class="practice-label" id="practice-speed-label">Speed 1.00×</span>
              <button type="button" class="ghost practice-nudge" id="btn-speed-down" aria-label="Slower">−</button>
              <input type="range" id="practice-speed" class="slider-speed" min="0.5" max="1.5" step="0.05" value="1" />
              <button type="button" class="ghost practice-nudge" id="btn-speed-up" aria-label="Faster">+</button>
            </div>
            <div class="practice-row">
              <span class="practice-label" id="practice-key-label">Key 0</span>
              <button type="button" class="ghost practice-nudge" id="btn-key-down" aria-label="Lower key">−</button>
              <input type="range" id="practice-key" class="slider-key" min="-6" max="6" step="1" value="0" />
              <button type="button" class="ghost practice-nudge" id="btn-key-up" aria-label="Raise key">+</button>
            </div>
            <button type="button" class="ghost practice-reset" id="btn-practice-reset">Reset</button>
            <button type="button" class="ghost" id="btn-listen-432" aria-pressed="false">Listen at 432 Hz</button>
            <label class="practice-label">Low shelf
              <input type="range" id="tone-low" min="-12" max="12" step="1" value="0" />
            </label>
            <label class="practice-label">High shelf
              <input type="range" id="tone-high" min="-12" max="12" step="1" value="0" />
            </label>
            <p class="practice-export-note">
              Save current mix includes Speed, Key, pan, mute/solo, and master.
              Save all tracks stays original speed/key.
            </p>
            <p class="practice-status" id="practice-status" hidden></p>
          </div>
        </div>
      </div>
      <div class="count-in-status" id="count-in-status" aria-live="polite" hidden></div>
      <div class="mixer-stage">
        <div class="mixer-video-wrap" id="mixer-video-wrap" hidden>
          <div class="mixer-video-toolbar">
            <div class="video-toolbar-group" role="group" aria-label="Place video">
              <button type="button" class="video-toolbar-btn mixer-video-drag" id="btn-drag-video" aria-label="Move video">Move</button>
              <button type="button" class="video-toolbar-btn" id="btn-dock-video" hidden>Dock</button>
              <button type="button" class="video-toolbar-btn" id="btn-hide-video">Hide video</button>
            </div>
            <span class="video-toolbar-divider" aria-hidden="true"></span>
            <div class="video-toolbar-group" role="group" aria-label="Video position">
              <button type="button" class="video-toolbar-btn video-snap-btn" data-snap="above" aria-pressed="true">Top</button>
              <button type="button" class="video-toolbar-btn video-snap-btn" data-snap="below" aria-pressed="false">Bottom</button>
              <button type="button" class="video-toolbar-btn video-snap-btn" data-snap="left" aria-pressed="false">Left</button>
              <button type="button" class="video-toolbar-btn video-snap-btn" data-snap="right" aria-pressed="false">Right</button>
            </div>
          </div>
          <div class="mixer-video" id="yt-player-slot"></div>
          <div class="mixer-video-resize" id="btn-resize-video" role="separator" aria-label="Resize video"></div>
        </div>
        <label class="master-vol">
          <span class="master-vol-label">Master</span>
          <span class="visually-hidden">Master volume in decibels</span>
          <input type="range" id="master-vol" class="vol-slider slider-master"
            min="${DB_MIN}" max="${DB_MAX}" step="0.5" value="${state.masterVolumeDb}" />
          <span id="master-vol-val" class="stem-db">${state.masterVolumeDb.toFixed(1)}</span>
        </label>
        <div class="mixer-stems-wrap">
          <div class="mixer-stems" id="stems"></div>
          <div class="mixer-stems-resize" id="btn-resize-stems" role="separator" aria-label="Resize stem waveforms" hidden></div>
        </div>
      </div>
      <details class="mixer-practice-panel" id="practice-panel" open>
        <summary>
          <span>Practice</span>
          <span class="practice-summary" id="practice-summary"></span>
        </summary>
        <div class="practice-panel-toolbar">
          <label class="practice-loop-toggle">
            <input type="checkbox" id="loop-toggle" />
            <span>Loop</span>
          </label>
          <button type="button" class="ghost practice-small" id="btn-loop-section">Loop current section</button>
          <button type="button" class="ghost practice-small" id="btn-loop-clear">Clear loop</button>
          <label class="practice-countin">
            <span>Count-in</span>
            <select id="count-in-bars">
              <option value="0">Off</option>
              <option value="1">1 bar</option>
              <option value="2">2 bars</option>
            </select>
          </label>
          <span class="practice-loop-status" id="loop-status" aria-live="polite"></span>
        </div>
        <div class="practice-panel-tabs" role="tablist" aria-label="Practice view">
          <button type="button" class="practice-tab active" data-practice-tab="chords" role="tab" aria-selected="true">Chords</button>
          <button type="button" class="practice-tab" data-practice-tab="sections" role="tab" aria-selected="false">Sections</button>
          <button type="button" class="practice-tab" data-practice-tab="lyrics" role="tab" aria-selected="false">Lyrics</button>
          <button type="button" class="practice-tab" data-practice-tab="tab" role="tab" aria-selected="false">Tab</button>
        </div>
        <div class="practice-panel-body" id="practice-body">
          <p class="practice-empty">No analysis in this run.</p>
        </div>
      </details>
    </div>
  `;

  const mixerRoot = root.querySelector<HTMLElement>(".mixer");
  mixerRoot?.addEventListener("pointerdown", () => {
    mixerRoot.focus({ preventScroll: true });
  });

  const stemsEl = document.getElementById("stems")!;
  const rowHtml = (stem: StemInfo, inGroup: boolean): string => {
      const db = clampDb(state.volumesDb[stem.id] ?? DB_DEFAULT);
      const pan = clampPan(state.pan[stem.id]);
      const muted = !!state.muted[stem.id];
      const soloed = !!state.soloed[stem.id];
      const liveMetro = stem.id === "metronome" && metroConfig != null;
      const wave = waveformSeekHtml(stem.peaks, stem.label, {
        variant: liveMetro ? "metro" : "default",
      });
      const download = stem.downloadUrl
        ? `<a class="download-link" href="${stem.downloadUrl}" download="${escapeHtml(
            stem.downloadFilename || `${stem.label}.wav`
          )}">Download</a>`
        : "";
      const preview = liveMetro
        ? metronomePreviewHtml(wave, download)
        : wave || download
          ? `<div class="stem-preview">${wave}${download}</div>`
          : "";
      // Metronome: keep a short title tooltip only — no body hint clutter.
      const meterGain = effectiveGains(stemInfos.map((s) => s.id), state)[stem.id] ?? 0;
      const meterShown = stemMeterLevel(stem.peaks, meterGain, null);
      const nameTitle =
        stem.hint && !liveMetro ? ` title="${escapeHtml(stem.hint)}"` : "";
      const hint =
        stem.hint && !liveMetro
          ? `<span class="stem-hint">${escapeHtml(stem.hint)}</span>`
          : "";
      return `
        <div class="stem-row${liveMetro ? " is-metro" : ""}" data-id="${escapeHtml(stem.id)}">
          <button type="button" class="toggle mute${muted ? " active" : ""}" data-id="${escapeHtml(stem.id)}"
            aria-label="${muted ? "Unmute" : "Mute"} ${escapeHtml(stem.label)}"
            aria-pressed="${muted ? "true" : "false"}">M</button>
          <button type="button" class="toggle solo${soloed ? " active" : ""}" data-id="${escapeHtml(stem.id)}"
            aria-label="${soloed ? "Unsolo" : "Solo"} ${escapeHtml(stem.label)}"
            aria-pressed="${soloed ? "true" : "false"}">S</button>
          <div class="stem-name-block">
            <strong class="stem-name"${nameTitle}>${escapeHtml(
              inGroup ? stem.shortLabel || stem.label : stem.label
            )}</strong>
            <span class="stem-meter" style="--level:${meterShown}" role="meter" aria-label="${escapeHtml(stem.label)} level" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(meterShown * 100)}"></span>
            ${hint}
          </div>
          <label class="stem-vol">
            <span class="visually-hidden">${escapeHtml(stem.label)} volume in decibels</span>
            <input type="range" class="vol-slider" data-id="${escapeHtml(stem.id)}"
              min="${DB_MIN}" max="${DB_MAX}" step="0.5" value="${db}" />
            <span class="vol-val stem-db">${db.toFixed(1)}</span>
          </label>
          <label class="stem-pan" title="Pan: drag, use arrow keys, or double-click to center">
            <span class="pan-name" aria-hidden="true">Pan</span>
            <span class="visually-hidden">${escapeHtml(stem.label)} pan, left to right</span>
            <input type="range" class="pan-slider" data-id="${escapeHtml(stem.id)}"
              min="${PAN_MIN}" max="${PAN_MAX}" step="1" value="${pan}"
              aria-valuetext="${panSpoken(pan)}" />
            <span class="pan-val" aria-hidden="true">${formatPan(pan)}</span>
          </label>
          ${preview}
        </div>`;
  };
  stemsEl.innerHTML = stemListItems(stemInfos)
    .map((item, index) =>
      item.kind === "stem"
        ? rowHtml(item.stem, false)
        : groupHtml(item, index, (stem) => rowHtml(stem, true))
    )
    .join("");
  paintStemMeters(engine.currentTime(), engine.getDuration(), engine.isPlaying());
  bindGroupToggles(stemsEl);
  updateGroupSummaries();

  document.getElementById("btn-playpause")!.onpointerdown = (event) => {
    event.preventDefault();
    togglePlayPause();
  };
  document.getElementById("btn-stop")!.onpointerdown = (event) => {
    event.preventDefault();
    stopPlayback();
  };
  document.getElementById("btn-restart")!.onclick = () => {
    wantPlaying = true;
    setPlayPauseLabel(true);
    void engine.restart().then(() => {
      nudgeYoutubeToAudio();
      if (engine.isPlaying()) clearSleepResumeHint();
      saveTransport();
    });
  };

  document.getElementById("btn-muteall")!.onclick = () => {
    // If every stem is already muted, Unmute All; otherwise Mute All.
    const target = !allMuted();
    activePreset = "custom";
    for (const stem of stemInfos) {
      state.muted[stem.id] = target;
      // Mute and solo are mutually exclusive per stem.
      if (target) state.soloed[stem.id] = false;
    }
    applyStateGains();
    reportState();
    updateStemChrome();
    updateMuteAllLabel();
  };

  document.getElementById("btn-reset")!.onclick = () => {
    for (const stem of stemInfos) {
      state.muted[stem.id] = stem.id === "metronome";
      state.soloed[stem.id] = false;
      state.volumesDb[stem.id] = DB_DEFAULT;
      state.pan[stem.id] = PAN_DEFAULT;
    }
    state.masterVolumeDb = DB_DEFAULT;
    activePreset = "all";
    stemsEl.querySelectorAll<HTMLInputElement>(".vol-slider").forEach((el) => {
      el.value = String(DB_DEFAULT);
      const label = el.parentElement?.querySelector(".vol-val");
      if (label) label.textContent = DB_DEFAULT.toFixed(1);
    });
    stemsEl.querySelectorAll<HTMLInputElement>(".pan-slider").forEach((el) => {
      paintPan(el, PAN_DEFAULT);
    });
    const masterEl = document.getElementById("master-vol") as HTMLInputElement | null;
    const masterVal = document.getElementById("master-vol-val");
    if (masterEl) {
      masterEl.value = String(DB_DEFAULT);
      syncRangeFill(masterEl);
    }
    if (masterVal) masterVal.textContent = DB_DEFAULT.toFixed(1);
    applyStateGains();
    reportState();
    updateStemChrome();
    updateMuteAllLabel();
  };

  document.getElementById("btn-clearsolo")!.onclick = () => {
    for (const stem of stemInfos) {
      state.soloed[stem.id] = false;
    }
    activePreset = "all";
    applyStateGains();
    reportState();
    updateStemChrome();
    updateMuteAllLabel();
  };

  document.querySelectorAll<HTMLButtonElement>(".mixer-presets button").forEach((el) => {
    el.onclick = () => {
      const preset = el.dataset.preset as Exclude<PlaybackPresetId, "custom">;
      const soloed = presetSoloMap(stemInfos.map((s) => s.id), preset);
      if (!soloed) return;
      state.soloed = soloed;
      activePreset = preset;
      applyStateGains();
      reportState();
      updateStemChrome();
      updateMuteAllLabel();
    };
  });

  document.getElementById("btn-dock-video")!.onclick = () => {
    videoLayout = dockedVideoLayout(videoLayout);
    videoLayout.placed = false;
    applyVideoLayout();
    saveVideoLayout();
    reportState(true);
  };
  document.querySelectorAll<HTMLButtonElement>(".mixer-video-toolbar [data-snap]").forEach((el) => {
    el.onclick = () => {
      const snap = videoSnap(el.dataset.snap);
      videoLayout = dockedVideoLayout(videoLayout);
      videoLayout.placed = false;
      videoLayout.snap = snap;
      applyVideoLayout();
      saveVideoLayout();
      reportState(true);
    };
  });

  document.getElementById("btn-hide-video")!.onclick = () => {
    hideYoutubeVideo = true;
    destroyYoutubePlayer();
    destroyLocalVideo();
    updateYoutubeChrome();
    reportState(true);
  };
  document.getElementById("btn-show-video")!.onclick = () => {
    hideYoutubeVideo = false;
    updateYoutubeChrome();
    applyVideoLayout();
    mountPicture();
    scheduleFrameHeight();
    reportState(true);
  };
  bindVideoFrameGestures();
  bindStemWaveGestures();
  updateMuteAllLabel();
  updateStemChrome();
  updateYoutubeChrome();
  applyStemWaveHeight();

  const seek = document.getElementById("seek") as HTMLInputElement;
  syncRangeFill(seek);
  seek.oninput = () => {
    syncRangeFill(seek);
    const dur = engine.getDuration() || 1;
    const t = (Number(seek.value) / 1000) * dur;
    syncOrigin = "audio";
    void engine.seek(t).then(() => nudgeYoutubeToAudio());
    updatePlayheads(t, dur);
  };

  stemsEl.querySelectorAll<HTMLElement>(".wave-seek").forEach((el) => {
    el.onpointerdown = (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      seekFromWavePointer(el, event.clientX);
    };
  });

  const masterVol = document.getElementById("master-vol") as HTMLInputElement;
  syncRangeFill(masterVol);
  masterVol.oninput = () => {
    syncRangeFill(masterVol);
    const db = Number(masterVol.value);
    state.masterVolumeDb = db;
    const label = document.getElementById("master-vol-val");
    if (label) label.textContent = db.toFixed(1);
    applyStateGains();
  };
  masterVol.onchange = () => {
    reportState(true);
  };

  bindPracticeControls();

  stemsEl.querySelectorAll<HTMLInputElement>(".vol-slider").forEach((el) => {
    el.oninput = () => {
      const id = el.dataset.id!;
      const db = Number(el.value);
      state.volumesDb[id] = db;
      const label = el.parentElement?.querySelector(".vol-val");
      if (label) label.textContent = db.toFixed(1);
      applyStateGains();
      // Live audio only while dragging — report to Streamlit on release.
    };
    el.onchange = () => {
      reportState(true);
    };
  });

  stemsEl.querySelectorAll<HTMLInputElement>(".pan-slider").forEach((el) => {
    const id = el.dataset.id!;
    // Mouse and touch drags stick to center; arrow keys step by one and never snap.
    let dragging = false;
    const endDrag = () => {
      dragging = false;
    };
    el.addEventListener("pointerdown", () => {
      dragging = true;
    });
    el.addEventListener("pointerup", endDrag);
    el.addEventListener("pointercancel", endDrag);
    el.addEventListener("keydown", endDrag);
    el.addEventListener("blur", endDrag);
    el.oninput = () => {
      const raw = clampPan(el.value);
      const pan = dragging ? snapPanToCenter(raw) : raw;
      state.pan[id] = pan;
      paintPan(el, pan);
      // Live audio while dragging; report to Streamlit on release.
      engine.applyPans(state.pan);
    };
    el.onchange = () => {
      reportState(true);
    };
    el.ondblclick = () => {
      state.pan[id] = PAN_DEFAULT;
      paintPan(el, PAN_DEFAULT);
      engine.applyPans(state.pan);
      reportState(true);
    };
  });

  stemsEl.querySelectorAll<HTMLButtonElement>("button.toggle.mute").forEach((el) => {
    el.onclick = () => {
      const id = el.dataset.id!;
      const muted = !state.muted[id];
      activePreset = "custom";
      state.muted[id] = muted;
      if (muted) state.soloed[id] = false;
      applyStateGains();
      reportState();
      updateStemChrome();
      updateMuteAllLabel();
    };
  });

  stemsEl.querySelectorAll<HTMLButtonElement>("button.toggle.solo").forEach((el) => {
    el.onclick = () => {
      const id = el.dataset.id!;
      const soloed = !state.soloed[id];
      activePreset = "custom";
      state.soloed[id] = soloed;
      if (soloed) state.muted[id] = false;
      applyStateGains();
      reportState();
      updateStemChrome();
      updateMuteAllLabel();
    };
  });

  bindMetronomeOptionControls();

  engine.setTimeCallback((t, dur, playing) => {
    const currentEl = document.getElementById("time-current");
    const durationEl = document.getElementById("time-duration");
    const seekEl = document.getElementById("seek") as HTMLInputElement | null;
    if (currentEl) currentEl.textContent = formatTime(t);
    if (durationEl) durationEl.textContent = formatTime(dur);
    if (seekEl) {
      seekEl.disabled = dur <= 0;
      seekEl.setAttribute("aria-valuetext", formatTime(t));
      if (document.activeElement !== seekEl) {
        seekEl.value = String(dur > 0 ? Math.round((t / dur) * 1000) : 0);
      }
      syncRangeFill(seekEl);
    }
    updatePlayheads(t, dur);
    paintStemMeters(t, dur, playing);
    postSidebarChord(t);
    updatePracticePanelAt(t);
    if (!hideYoutubeVideo && (ytPlayer || localVideoEl)) {
      syncPictureToStems(playing);
    }
    if (playing) {
      clearSleepResumeHint();
      publishMediaSession(true);
      updateMediaSessionPosition();
    } else if (!wantPlaying && !engine.isKeepPlaying()) {
      clearMediaSession();
    }
    if (!transportBusy && transportPending === null) {
      const hold = playing || engine.isKeepPlaying();
      wantPlaying = hold;
      setPlayPauseLabel(hold);
    }
    saveTransport();
  });

  updateTrackTitleDisplay();
  applyVideoLayout();
  bindPracticePanel();
  renderPracticePanel();
  bindLayoutResizeObserver();
  mountPicture();
  scheduleFrameHeight();
}

function updateTrackTitleDisplay(): void {
  const el = document.getElementById("track-title");
  if (!el) return;
  if (trackTitle.trim()) {
    el.textContent = trackTitle.trim();
    el.style.display = "block";
  } else {
    el.textContent = "";
    el.style.display = "none";
  }
}

function showPracticeStatus(message: string | null): void {
  const el = document.getElementById("practice-status");
  if (!el) return;
  if (message) {
    el.hidden = false;
    el.textContent = message;
  } else {
    el.hidden = true;
    el.textContent = "";
  }
}

function syncPracticeChrome(settings: PracticeSettings): void {
  const speed = document.getElementById("practice-speed") as HTMLInputElement | null;
  const key = document.getElementById("practice-key") as HTMLInputElement | null;
  const speedLabel = document.getElementById("practice-speed-label");
  const keyLabel = document.getElementById("practice-key-label");
  if (speed) {
    speed.value = String(settings.rate);
    syncRangeFill(speed);
  }
  if (key) {
    key.value = String(settings.semitones);
    syncRangeFill(key);
  }
  if (speedLabel) speedLabel.textContent = formatPracticeSpeedLabel(settings.rate);
  if (keyLabel) keyLabel.textContent = formatPracticeKeyLabel(settings.semitones);
}

function bindPracticeControls(): void {
  const speed = document.getElementById("practice-speed") as HTMLInputElement | null;
  const key = document.getElementById("practice-key") as HTMLInputElement | null;
  if (!speed || !key) return;

  const persisted = loadPracticeSettings();
  syncPracticeChrome(persisted);
  void engine.setPracticeRate(persisted.rate).then((r) => {
    if (!r.ok && r.message) showPracticeStatus(r.message);
    applyPicturePlaybackRate();
  });
  void engine.setPracticeSemitones(persisted.semitones).then((r) => {
    if (!r.ok && r.message) showPracticeStatus(r.message);
    syncPracticeChrome(engine.getPracticeSettings());
  });

  const persist = () => {
    const settings = engine.getPracticeSettings();
    savePracticeSettings(settings);
    syncPracticeChrome(settings);
    reportState(true);
  };

  const afterSpeedChange = () => {
    applyPicturePlaybackRate();
    if (!hideYoutubeVideo && (ytPlayer || localVideoEl)) {
      nudgeYoutubeToAudio();
    }
  };

  const applyRate = (raw: number) => {
    const next = clampPracticeRate(raw);
    void engine.setPracticeRate(next).then((r) => {
      if (!r.ok && r.message) showPracticeStatus(r.message);
      else showPracticeStatus(null);
      persist();
      afterSpeedChange();
    });
  };
  const applyKey = (raw: number) => {
    const next = clampPracticeSemitones(raw);
    void engine.setPracticeSemitones(next).then((r) => {
      if (!r.ok && r.message) showPracticeStatus(r.message);
      else showPracticeStatus(null);
      persist();
      // Key changes pitch only — do not change picture tempo.
    });
  };

  speed.oninput = () => {
    syncRangeFill(speed);
    applyRate(Number(speed.value));
  };
  key.oninput = () => {
    syncRangeFill(key);
    applyKey(Number(key.value));
  };
  document.getElementById("btn-speed-down")!.onclick = () =>
    applyRate(Number(speed.value) - PRACTICE_RATE_STEP);
  document.getElementById("btn-speed-up")!.onclick = () =>
    applyRate(Number(speed.value) + PRACTICE_RATE_STEP);
  document.getElementById("btn-key-down")!.onclick = () =>
    applyKey(Number(key.value) - 1);
  document.getElementById("btn-key-up")!.onclick = () =>
    applyKey(Number(key.value) + 1);
  document.getElementById("btn-practice-reset")!.onclick = () => {
    void engine.resetPractice().then((r) => {
      if (!r.ok && r.message) showPracticeStatus(r.message);
      else showPracticeStatus(null);
      persist();
      afterSpeedChange();
    });
  };
  const listen432 = document.getElementById("btn-listen-432");
  if (listen432) {
    listen432.setAttribute("aria-pressed", listenA432 ? "true" : "false");
    listen432.onclick = () => {
      listenA432 = !listenA432;
      engine.setListenHertz(listenA432 ? 432 : null);
      listen432.setAttribute("aria-pressed", listenA432 ? "true" : "false");
    };
  }
  const toneLow = document.getElementById("tone-low") as HTMLInputElement | null;
  const toneHigh = document.getElementById("tone-high") as HTMLInputElement | null;
  const applyTone = () => {
    engine.setShelves(Number(toneLow?.value ?? 0), Number(toneHigh?.value ?? 0));
  };
  if (toneLow) toneLow.oninput = applyTone;
  if (toneHigh) toneHigh.oninput = applyTone;
  const speedKeyBtn = document.getElementById("btn-speed-key");
  const speedPopover = document.getElementById("practice-speed-popover");
  const mixBtn = document.getElementById("btn-mix");
  const mixMenu = document.getElementById("mix-menu");
  if (speedKeyBtn && speedPopover && mixBtn && mixMenu) {
    speedKeyBtn.onclick = () => {
      const opening = speedPopover.hidden;
      if (opening) {
        mixMenu.hidden = true;
        mixBtn.setAttribute("aria-expanded", "false");
      }
      speedPopover.hidden = !opening;
      speedKeyBtn.setAttribute("aria-expanded", opening ? "true" : "false");
      syncMixerMenuBackdrop();
    };
    mixBtn.onclick = () => {
      const opening = mixMenu.hidden;
      if (opening) {
        speedPopover.hidden = true;
        speedKeyBtn.setAttribute("aria-expanded", "false");
      }
      mixMenu.hidden = !opening;
      mixBtn.setAttribute("aria-expanded", opening ? "true" : "false");
      syncMixerMenuBackdrop();
    };
  }
  const sheetBtn = document.getElementById("btn-practice-sheet");
  const sheet = document.getElementById("practice-panel");
  if (sheetBtn && sheet) {
    if (practiceSheetOpen === null) {
      practiceSheetOpen = !window.matchMedia("(max-width: 900px)").matches;
    }
    sheet.hidden = !practiceSheetOpen;
    sheetBtn.setAttribute("aria-expanded", practiceSheetOpen ? "true" : "false");
    sheetBtn.onclick = () => {
      const opening = sheet.hidden;
      sheet.hidden = !opening;
      practiceSheetOpen = opening;
      sheetBtn.setAttribute("aria-expanded", opening ? "true" : "false");
    };
  }
  document.removeEventListener("keydown", onSpeedPopoverEscape);
  document.addEventListener("keydown", onSpeedPopoverEscape);
  document.removeEventListener("pointerdown", onMixerMenuPointerDown, true);
  document.addEventListener("pointerdown", onMixerMenuPointerDown, true);
  window.removeEventListener("blur", onMixerMenuWindowBlur);
  window.addEventListener("blur", onMixerMenuWindowBlur);
  const practiceGroup = document.querySelector(".mixer-practice");
  practiceGroup?.addEventListener("dblclick", (ev) => {
    if ((ev.target as HTMLElement).closest("button, input")) return;
    void engine.resetPractice().then(() => {
      showPracticeStatus(null);
      persist();
      afterSpeedChange();
    });
  });
}

function allMuted(): boolean {
  return stemInfos.length > 0 && stemInfos.every((s) => !!state.muted[s.id]);
}

function anySoloed(): boolean {
  return stemInfos.some((s) => !!state.soloed[s.id]);
}

function isAudible(stemId: string): boolean {
  return anySoloed() ? !!state.soloed[stemId] : !state.muted[stemId];
}

function updateMuteAllLabel(): void {
  const btn = document.getElementById("btn-muteall");
  if (btn) btn.textContent = allMuted() ? "Unmute all" : "Mute all";
}

function updatePresetChrome(): void {
  const ids = stemInfos.map((s) => s.id);
  document.querySelectorAll<HTMLButtonElement>(".mixer-presets button").forEach((el) => {
    const preset = el.dataset.preset as Exclude<PlaybackPresetId, "custom">;
    const available = presetSoloMap(ids, preset) !== null;
    el.disabled = !available;
    const on = available && activePreset === preset;
    el.classList.toggle("active", on);
    el.setAttribute("aria-pressed", on ? "true" : "false");
  });
}

type StemListItem =
  | { kind: "stem"; stem: StemInfo }
  | { kind: "group"; key: string; label: string; members: StemInfo[] };

/**
 * Stems from "Break down a stem" share a parent. Two or more of them fold under one
 * disclosure header, placed where the first one was. A lone child stays a plain row.
 * A group is only a way to hide rows: it has no fader and no sum, and hiding never mutes.
 */
function stemListItems(stems: StemInfo[]): StemListItem[] {
  const byGroup = new Map<string, StemInfo[]>();
  for (const s of stems) {
    if (s.group) byGroup.set(s.group, [...(byGroup.get(s.group) ?? []), s]);
  }
  const items: StemListItem[] = [];
  const emitted = new Set<string>();
  for (const stem of stems) {
    const members = stem.group ? byGroup.get(stem.group) : undefined;
    if (!stem.group || !members || members.length < 2) {
      items.push({ kind: "stem", stem });
    } else if (!emitted.has(stem.group)) {
      emitted.add(stem.group);
      items.push({
        kind: "group",
        key: stem.group,
        label: stem.groupLabel || stem.group,
        members,
      });
    }
  }
  return items;
}

function groupHtml(
  item: Extract<StemListItem, { kind: "group" }>,
  index: number,
  rowFor: (stem: StemInfo) => string
): string {
  const open = !collapsedGroups.has(item.key);
  const bodyId = `group-rows-${index}`;
  const members = escapeHtml(JSON.stringify(item.members.map((m) => m.id)));
  const label = escapeHtml(item.label);
  return `
    <div class="stem-group" data-group="${escapeHtml(item.key)}" data-members="${members}">
      <button type="button" class="group-toggle" aria-expanded="${open}" aria-controls="${bodyId}"
        title="Show or hide these stems. Hidden stems keep playing with their mute, solo, level, and pan.">
        <span class="group-caret" aria-hidden="true"></span>
        <span class="group-name">${label}</span>
        <span class="group-meta"></span>
      </button>
      <div class="stem-group-rows" id="${bodyId}" role="group" aria-label="${label} stems"${
        open ? "" : " hidden"
      }>
        ${item.members.map(rowFor).join("")}
      </div>
    </div>`;
}

function bindGroupToggles(root: HTMLElement): void {
  root.querySelectorAll<HTMLButtonElement>(".group-toggle").forEach((btn) => {
    btn.onclick = () => {
      const wrap = btn.closest<HTMLElement>(".stem-group");
      const key = wrap?.dataset.group;
      const body = wrap?.querySelector<HTMLElement>(".stem-group-rows");
      if (!key || !body) return;
      const open = btn.getAttribute("aria-expanded") !== "true";
      if (open) collapsedGroups.delete(key);
      else collapsedGroups.add(key);
      btn.setAttribute("aria-expanded", open ? "true" : "false");
      body.hidden = !open;
      scheduleFrameHeight();
    };
  });
}

/** "3 stems · 1 muted · 1 solo": hidden rows still say what they are doing. */
function updateGroupSummaries(): void {
  document.querySelectorAll<HTMLElement>(".stem-group").forEach((el) => {
    let ids: string[] = [];
    try {
      ids = JSON.parse(el.dataset.members ?? "[]") as string[];
    } catch {
      /* summary stays empty */
    }
    const muted = ids.filter((id) => state.muted[id]).length;
    const soloed = ids.filter((id) => state.soloed[id]).length;
    const parts = [`${ids.length} stems`];
    if (muted) parts.push(`${muted} muted`);
    if (soloed) parts.push(`${soloed} solo`);
    const meta = el.querySelector(".group-meta");
    if (meta) meta.textContent = parts.join(" · ");
  });
}

function updateStemChrome(): void {
  const clearBtn = document.getElementById("btn-clearsolo");
  if (clearBtn) clearBtn.hidden = !anySoloed();
  updatePresetChrome();
  updateGroupSummaries();

  document.querySelectorAll<HTMLElement>(".stem-row").forEach((row) => {
    const id = row.dataset.id!;
    const stem = stemInfos.find((s) => s.id === id);
    const label = stem?.label ?? id;
    const muted = !!state.muted[id];
    const soloed = !!state.soloed[id];
    const audible = isAudible(id);
    const silencedBySolo = !audible && !muted && !soloed;
    row.classList.toggle("silent", !audible);
    if (silencedBySolo) {
      row.title = "Silent because another stem is soloed";
    } else if (stem?.hint) {
      row.title = stem.hint;
    } else {
      row.removeAttribute("title");
    }

    const muteBtn = row.querySelector<HTMLButtonElement>("button.toggle.mute");
    if (muteBtn) {
      muteBtn.classList.toggle("active", muted);
      muteBtn.setAttribute("aria-pressed", muted ? "true" : "false");
      muteBtn.setAttribute("aria-label", `${muted ? "Unmute" : "Mute"} ${label}`);
    }
    const soloBtn = row.querySelector<HTMLButtonElement>("button.toggle.solo");
    if (soloBtn) {
      soloBtn.classList.toggle("active", soloed);
      soloBtn.setAttribute("aria-pressed", soloed ? "true" : "false");
      soloBtn.setAttribute("aria-label", `${soloed ? "Unsolo" : "Solo"} ${label}`);
    }

    const nameEl = row.querySelector(".stem-name");
    if (nameEl) {
      const hidden = nameEl.querySelector(".visually-hidden");
      if (silencedBySolo && !hidden) {
        const note = document.createElement("span");
        note.className = "visually-hidden";
        note.textContent = " Silent because another stem is soloed";
        nameEl.appendChild(note);
      } else if (!silencedBySolo && hidden) {
        hidden.remove();
      }
    }
  });
  scheduleFrameHeight();
}

function updatePlayheads(t: number, dur: number): void {
  const pct = dur > 0 ? Math.min(100, Math.max(0, (t / dur) * 100)) : 0;
  document.querySelectorAll<HTMLElement>(".wave-playhead").forEach((el) => {
    el.style.left = `${pct}%`;
    el.hidden = dur <= 0;
  });
  document.querySelectorAll<HTMLElement>(".wave-seek").forEach((el) => {
    el.setAttribute("aria-valuenow", String(Math.round(pct * 10)));
    el.setAttribute("aria-valuetext", formatTime(t));
  });
}

function seekFromWavePointer(el: HTMLElement, clientX: number): void {
  const dur = engine.getDuration();
  if (!(dur > 0)) return;
  const rect = el.getBoundingClientRect();
  const ratio = rect.width > 0 ? (clientX - rect.left) / rect.width : 0;
  const t = Math.max(0, Math.min(1, ratio)) * dur;
  syncOrigin = "audio";
  void engine.seek(t).then(() => nudgeYoutubeToAudio());
  updatePlayheads(t, dur);
}

function peakPercentile(peaks: number[], p: number): number {
  if (peaks.length === 0) return 0;
  const sorted = [...peaks].sort((a, b) => a - b);
  const idx = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil(p * sorted.length) - 1)
  );
  return sorted[idx];
}

function stemMeterLevel(
  peaks: number[] | undefined,
  gain: number,
  sampleIndex: number | null
): number {
  if (!peaks || peaks.length === 0) return 0;
  if (!Number.isFinite(gain) || gain <= 0) return 0;
  let amplitude = 0;
  if (sampleIndex === null) {
    for (const value of peaks) amplitude = Math.max(amplitude, Math.abs(Number(value) || 0));
  } else {
    amplitude = Math.abs(Number(peaks[sampleIndex]) || 0);
  }
  return Math.max(0, Math.min(1, amplitude * gain));
}

function meterSampleIndex(
  peaksLength: number,
  t: number,
  dur: number,
  playing: boolean
): number | null {
  if (!playing || !(dur > 0) || !(peaksLength > 0)) return null;
  return Math.min(peaksLength - 1, Math.max(0, Math.floor((t / dur) * peaksLength)));
}

function paintStemMeters(t: number, dur: number, playing: boolean): void {
  const gains = effectiveGains(
    stemInfos.map((s) => s.id),
    state
  );
  document.querySelectorAll<HTMLElement>(".stem-row").forEach((row) => {
    const id = row.dataset.id;
    if (!id) return;
    const meter = row.querySelector<HTMLElement>(".stem-meter");
    if (!meter) return;
    const stem = stemInfos.find((s) => s.id === id);
    const peaks = stem?.peaks;
    const index = meterSampleIndex(peaks?.length ?? 0, t, dur, playing);
    const level = stemMeterLevel(peaks, gains[id] ?? 0, index);
    meter.style.setProperty("--level", String(level));
    meter.setAttribute("aria-valuenow", String(Math.round(level * 100)));
  });
}

function waveformNorm(peaks: number[]): number {
  const absMax = Math.max(...peaks, 0.0001);
  const p98 = peakPercentile(peaks, 0.98);
  return Math.max(p98, absMax * 0.05, 0.0001);
}

/** Peak envelope from the same AudioBuffer used for playback. */
function peaksFromBuffer(buf: AudioBuffer, numPoints = 512): number[] {
  const nCh = buf.numberOfChannels;
  const length = buf.length;
  if (length <= 0 || nCh <= 0) return new Array(numPoints).fill(0);
  const chans: Float32Array[] = [];
  for (let c = 0; c < nCh; c++) chans.push(buf.getChannelData(c));
  const peakAt = (i: number): number => {
    let m = 0;
    for (const ch of chans) {
      const v = Math.abs(ch[i] ?? 0);
      if (v > m) m = v;
    }
    return m;
  };
  if (length <= numPoints) {
    const out = new Array(length);
    for (let i = 0; i < length; i++) out[i] = peakAt(i);
    return out;
  }
  const chunk = Math.max(1, Math.floor(length / numPoints));
  const peaks = new Array(numPoints);
  for (let i = 0; i < numPoints; i++) {
    const start = i * chunk;
    const end = i === numPoints - 1 ? length : Math.min(length, start + chunk);
    let m = 0;
    for (let s = start; s < end; s++) {
      const v = peakAt(s);
      if (v > m) m = v;
    }
    peaks[i] = m;
  }
  return peaks;
}

function paintDecodedWaves(): void {
  for (const stem of stemInfos) {
    const row = document.querySelector<HTMLElement>(
      `.stem-row[data-id="${cssAttr(stem.id)}"]`
    );
    if (!row) continue;
    const liveMetro = stem.id === "metronome" && metroConfig != null;
    const wave = waveformSeekHtml(stem.peaks, stem.label, {
      variant: liveMetro ? "metro" : "default",
    });
    if (!wave) continue;
    const metroSlot = liveMetro ? row.querySelector<HTMLElement>(".metro-wave") : null;
    if (metroSlot) {
      metroSlot.querySelector(".wave-seek")?.remove();
      metroSlot.insertAdjacentHTML("beforeend", wave);
      continue;
    }
    let preview = row.querySelector<HTMLElement>(".stem-preview");
    if (!preview) {
      preview = document.createElement("div");
      preview.className = "stem-preview";
      row.appendChild(preview);
    }
    preview.querySelector(".wave-seek")?.remove();
    preview.insertAdjacentHTML("afterbegin", wave);
  }
  document.querySelectorAll<HTMLElement>(".wave-seek").forEach((el) => {
    el.onpointerdown = (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      seekFromWavePointer(el, event.clientX);
    };
  });
}

function cssAttr(value: string): string {
  if (typeof CSS !== "undefined" && typeof CSS.escape === "function") return CSS.escape(value);
  return value.replace(/["\\]/g, "\\$&");
}

function setLoadProgress(loaded: number, total: number, message: string): void {
  const el = document.getElementById("status");
  if (el) el.textContent = message || `${loaded}/${total}`;
  const track = document.getElementById("mixer-load");
  const bar = document.getElementById("mixer-load-bar");
  if (!track || !bar) return;
  const ratio = total > 0 ? Math.max(0, Math.min(1, loaded / total)) : 0;
  track.hidden = false;
  bar.style.width = `${Math.round(ratio * 100)}%`;
}

function finishLoadProgress(message: string): void {
  const el = document.getElementById("status");
  if (el) el.textContent = message;
  const track = document.getElementById("mixer-load");
  const bar = document.getElementById("mixer-load-bar");
  if (bar) bar.style.width = "100%";
  const reduce =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce) {
    if (track) track.hidden = true;
    return;
  }
  window.setTimeout(() => {
    if (track) track.hidden = true;
  }, 220);
}

function applyDecodedPeaks(): void {
  for (const stem of stemInfos) {
    if (stem.id === "metronome" && metroConfig) {
      stem.peaks = metronomePeaksFromTimes(
        metronomeClickTimes(metroConfig, metroOptions.follow),
        metroConfig.durationSec || engine.getDuration(),
        metroOptions,
        metroConfig.refSec
      );
      continue;
    }
    const buf = engine.getBuffer(stem.id);
    if (!buf) continue;
    stem.peaks = peaksFromBuffer(buf);
  }
}

function metronomeToolbarHtml(): string {
  const rateBtns = METRONOME_RATE_CHOICES.map((r) => {
    const active = Math.abs(metroOptions.rate - r) < 1e-9;
    return `<button type="button" class="metro-rate-btn${active ? " active" : ""}" data-rate="${r}" aria-pressed="${active ? "true" : "false"}">${r}x</button>`;
  }).join("");
  const soundIds = Object.keys(METRONOME_SOUND_LABELS) as MetronomeSoundId[];
  const soundOpts = soundIds
    .map(
      (id) =>
        `<option value="${id}"${metroOptions.sound === id ? " selected" : ""}>${escapeHtml(
          METRONOME_SOUND_LABELS[id]
        )}</option>`
    )
    .join("");
  return `
    <div class="metro-toolbar" role="group" aria-label="Click track options">
      <div class="metro-rate-seg" role="group" aria-label="Click rate">${rateBtns}</div>
      <div class="metro-rate-seg" role="group" aria-label="Metronome mode">
        <button type="button" class="metro-follow-btn${metroOptions.follow === "smart" ? " active" : ""}" data-follow="smart" aria-pressed="${metroOptions.follow === "smart" ? "true" : "false"}" title="Follows the song's tempo changes">Smart</button>
        <button type="button" class="metro-follow-btn${metroOptions.follow === "steady" ? " active" : ""}" data-follow="steady" aria-pressed="${metroOptions.follow === "steady" ? "true" : "false"}" title="Even clicks at one tempo">Steady</button>
      </div>
      <button type="button" class="metro-chip${metroOptions.accent ? " active" : ""}" id="metro-accent"
        aria-pressed="${metroOptions.accent ? "true" : "false"}">Accent</button>
      <label class="metro-sound-wrap">
        <span class="visually-hidden">Click sound</span>
        <select id="metro-sound" aria-label="Click sound">${soundOpts}</select>
      </label>
    </div>`;
}

function metronomePreviewHtml(wave: string, download: string): string {
  return `<div class="stem-preview metro-preview">
    <div class="metro-wave">
      ${metronomeToolbarHtml()}
      ${wave}
    </div>
    ${download}
  </div>`;
}

function applyLiveMetronomeBuffer(): void {
  if (!metroConfig) return;
  const ctx = engine.createContextForDecode();
  const buf = buildMetronomeAudioBuffer(ctx, metroConfig, metroOptions);
  engine.replaceStemBuffer("metronome", buf);
  applyDecodedPeaks();
  const row = document.querySelector<HTMLElement>('.stem-row[data-id="metronome"]');
  if (!row) return;
  const preview = row.querySelector(".stem-preview");
  const stem = stemInfos.find((s) => s.id === "metronome");
  if (!stem || !preview) return;
  const wave = waveformSeekHtml(stem.peaks, stem.label, { variant: "metro" });
  const download = stem.downloadUrl
    ? `<a class="download-link" href="${stem.downloadUrl}" download="${escapeHtml(
        stem.downloadFilename || `${stem.label}.wav`
      )}">Download</a>`
    : "";
  preview.outerHTML = metronomePreviewHtml(wave, download);
  const nextPreview = row.querySelector(".stem-preview");
  nextPreview?.querySelectorAll<HTMLElement>(".wave-seek").forEach((el) => {
    el.onpointerdown = (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      seekFromWavePointer(el, event.clientX);
    };
  });
  bindMetronomeOptionControls();
  updatePlayheads(engine.currentTime(), engine.getDuration());
  scheduleFrameHeight();
}

function bindMetronomeOptionControls(): void {
  if (!metroConfig) return;
  const apply = (next: Partial<MetronomeOptions>) => {
    metroOptions = coerceMetronomeOptions({ ...metroOptions, ...next });
    engine.setCountInGrid(
      analysisData?.bpm,
      analysisData?.beats_per_measure,
      metroOptions
    );
    applyLiveMetronomeBuffer();
    applyStateGains();
    reportState(true);
  };
  document.querySelectorAll<HTMLButtonElement>(".metro-rate-btn").forEach((btn) => {
    btn.onclick = (event) => {
      event.stopPropagation();
      apply({ rate: coerceMetronomeRate(btn.dataset.rate) });
    };
  });
  document.querySelectorAll<HTMLButtonElement>(".metro-follow-btn").forEach((btn) => {
    btn.onclick = (event) => {
      event.stopPropagation();
      apply({ follow: coerceMetronomeFollow(btn.dataset.follow) });
    };
  });
  const accent = document.getElementById("metro-accent") as HTMLButtonElement | null;
  if (accent) {
    accent.onclick = (event) => {
      event.stopPropagation();
      apply({ accent: !metroOptions.accent });
    };
  }
  const sound = document.getElementById("metro-sound") as HTMLSelectElement | null;
  if (sound) {
    sound.onchange = () => {
      apply({ sound: coerceMetronomeSound(sound.value) });
    };
    sound.onpointerdown = (event) => event.stopPropagation();
  }
}

/** Peak i of n maps to the same 0–100% axis as the playhead (slot = width / n). */
function waveformBarLayout(n: number, width: number): { slot: number; barW: number } {
  const slot = width / Math.max(1, n);
  const barW = slot * 0.85;
  return { slot, barW };
}

function waveformSeekHtml(
  peaks: number[] | undefined,
  label: string,
  opts?: { variant?: "default" | "metro" }
): string {
  if (!peaks || peaks.length === 0) return "";
  const width = 400;
  const height = opts?.variant === "metro" ? 44 : 52;
  const mid = height / 2;
  const n = peaks.length;
  const norm = waveformNorm(peaks);
  const { slot, barW } = waveformBarLayout(n, width);
  const bars: string[] = [
    `<line class="wave-baseline" x1="0" y1="${mid}" x2="${width}" y2="${mid}" />`,
  ];
  const metro = opts?.variant === "metro";
  for (let i = 0; i < n; i++) {
    const peak = peaks[i] ?? 0;
    if (peak <= 0) continue;
    const amp = Math.sqrt(Math.min(1, peak / norm)) * (mid - 2);
    const h = Math.max(metro ? 4 : 3, amp * 2 * (metro ? 0.85 : 1));
    const x = i * slot + (slot - (metro ? Math.min(barW, 1.6) : barW)) / 2;
    const y = mid - h / 2;
    const w = metro ? Math.min(barW, 1.6) : barW;
    bars.push(
      `<rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${w.toFixed(2)}" ` +
        `height="${h.toFixed(2)}" rx="0.8" ry="0.8" />`
    );
  }
  const svg =
    `<svg class="waveform-svg" viewBox="0 0 ${width} ${height}" ` +
    `preserveAspectRatio="none" aria-hidden="true">${bars.join("")}</svg>`;
  return (
    `<div class="wave-seek" role="slider" tabindex="0" ` +
    `aria-label="Seek ${escapeHtml(label)}" aria-valuemin="0" aria-valuemax="1000" aria-valuenow="0">` +
    `${svg}<span class="wave-playhead" aria-hidden="true" hidden></span></div>`
  );
}

function isTypingTarget(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  const tag = el.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
  if (el.isContentEditable) return true;
  return false;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function stemKey(stems: StemInfo[], metro: MetronomeConfig | null): string {
  const stemPart = stems
    .map((s) => {
      if (s.id === "metronome" && metro && metro.times1x.length > 0) {
        return `${s.id}|live`;
      }
      return `${s.id}|${s.url}`;
    })
    .join(";");
  if (!metro || metro.times1x.length === 0) return stemPart;
  const first = metro.times1x[0] ?? 0;
  const last = metro.times1x[metro.times1x.length - 1] ?? 0;
  return `${stemPart}|m:${metro.times1x.length}:${first.toFixed(3)}:${last.toFixed(3)}:${metro.durationSec.toFixed(3)}`;
}

function parseMetronomeArg(raw: unknown): MetronomeConfig | null {
  if (!raw || typeof raw !== "object") return null;
  const obj = raw as Record<string, unknown>;
  const timesRaw = obj.times1x ?? obj.click_times_1x;
  const times1x = Array.isArray(timesRaw)
    ? timesRaw.map((t) => Number(t)).filter((t) => Number.isFinite(t))
    : [];
  if (times1x.length === 0) return null;
  const steadyRaw = obj.steadyTimes1x ?? obj.steady_times_1x;
  const steadyTimes1x = Array.isArray(steadyRaw)
    ? steadyRaw.map((t) => Number(t)).filter((t) => Number.isFinite(t))
    : [];
  const options = coerceMetronomeOptions(
    (obj.options ??
      {
        accent: obj.accent as boolean | undefined,
        rate: obj.rate as number | undefined,
        sound: obj.sound as MetronomeSoundId | undefined,
      }) as Partial<MetronomeOptions>
  );
  return {
    times1x,
    steadyTimes1x,
    refSec: Number(obj.refSec ?? obj.body_start_sec ?? obj.first_detected_sec ?? times1x[0] ?? 0) || 0,
    beatsPerMeasure: Math.max(1, Math.floor(Number(obj.beatsPerMeasure ?? obj.beats_per_measure ?? 4) || 4)),
    durationSec: Math.max(0.05, Number(obj.durationSec ?? obj.duration_sec ?? 0) || 0),
    options,
  };
}

function softUpdateStemDownloads(stems: StemInfo[]): void {
  stemInfos = stems.map((next) => {
    const prev = stemInfos.find((s) => s.id === next.id);
    return {
      ...next,
      peaks: prev?.peaks ?? next.peaks,
    };
  });
  for (const stem of stemInfos) {
    const row = document.querySelector(`.stem-row[data-id="${CSS.escape(stem.id)}"]`);
    if (!row || !stem.downloadUrl) continue;
    const link = row.querySelector<HTMLAnchorElement>("a.download-link");
    if (!link) continue;
    link.href = stem.downloadUrl;
    if (stem.downloadFilename) {
      link.setAttribute("download", stem.downloadFilename);
    }
  }
}

async function onRender(event: Event): Promise<void> {
  const data = (event as CustomEvent<RenderData>).detail;
  const args = data.args as {
    stems?: StemInfo[];
    initialVolumesDb?: Record<string, number>;
    initialPan?: Record<string, number>;
    initialMuted?: Record<string, boolean>;
    initialSoloed?: Record<string, boolean>;
    initialMasterVolumeDb?: number;
    trackTitle?: string;
    metronome?: unknown;
    youtubeVideoId?: string;
    hideYoutubeVideo?: boolean;
    localVideoUrl?: string;
    videoOffsetSec?: number;
    analysis?: unknown;
    tab?: unknown;
    initialLoop?: { start?: number; end?: number } | null;
    countInBars?: number;
    initialRate?: number;
    initialSemitones?: number;
    stemTrims?: Record<string, { start?: number; end?: number }> | null;
    seekSec?: number | null;
  };

  const stems = args.stems ?? [];
  const nextMetro = parseMetronomeArg(args.metronome);
  const key = stemKey(stems, nextMetro);
  const status = () => document.getElementById("status");
  const nextTitle = args.trackTitle ?? "";
  if (nextTitle !== trackTitle) {
    trackTitle = nextTitle;
    updateTrackTitleDisplay();
  }

  if (key === lastStemKey) {
    softUpdateStemDownloads(stems);
    applyYoutubeArgs(
      args.youtubeVideoId ?? "",
      Boolean(args.hideYoutubeVideo),
      args.localVideoUrl ?? "",
      Number(args.videoOffsetSec) || 0
    );
    applyPracticeArgs(args);
    scheduleFrameHeight();
    return;
  }

  try {
    const raw = sessionStorage.getItem(TRANSPORT_STORAGE_KEY);
    if (raw) {
      const stored = JSON.parse(raw) as { stemKey?: string };
      if (stored.stemKey !== key) {
        sessionStorage.removeItem(TRANSPORT_STORAGE_KEY);
      }
    }
  } catch {
    /* ignore */
  }
  lastStemKey = key;
  lastPracticeLoopArg = null;
  lastPracticeCountInArg = null;
  collapsedGroups.clear();
  wantPlaying = false;
  transportBusy = false;
  transportPending = null;
  stemInfos = stems;
  metroConfig = nextMetro;
  metroOptions = nextMetro
    ? coerceMetronomeOptions(nextMetro.options)
    : coerceMetronomeOptions({ accent: true, rate: 1, sound: "classic", follow: "smart" });
  const masterInit = Number(args.initialMasterVolumeDb);
  const volumesIn = { ...(args.initialVolumesDb ?? {}) };
  for (const id of Object.keys(volumesIn)) {
    volumesIn[id] = clampDb(Number(volumesIn[id]));
  }
  state = {
    volumesDb: volumesIn,
    pan: Object.fromEntries(
      stems.map((s) => [s.id, clampPan((args.initialPan ?? {})[s.id])])
    ),
    muted: { ...(args.initialMuted ?? {}) },
    soloed: { ...(args.initialSoloed ?? {}) },
    masterVolumeDb: Number.isFinite(masterInit) ? clampDb(masterInit) : DB_DEFAULT,
  };
  for (const s of stems) {
    if (state.volumesDb[s.id] === undefined) state.volumesDb[s.id] = DB_DEFAULT;
    else state.volumesDb[s.id] = clampDb(state.volumesDb[s.id]);
    if (state.muted[s.id] === undefined) state.muted[s.id] = s.id === "metronome";
    if (state.soloed[s.id] === undefined) state.soloed[s.id] = false;
  }
  activePreset = "all";
  destroyYoutubePlayer();
  destroyLocalVideo();
  youtubeVideoId = "";
  localVideoUrl = "";
  localVideoFailedUrl = "";
  hideYoutubeVideo = true;
  analysisData = adoptAnalysis(parseAnalysisArg(args.analysis));
  tabData = parseTabArg(args.tab);
  engine.setStemTrims(args.stemTrims);
  if (typeof args.seekSec === "number" && Number.isFinite(args.seekSec) && args.seekSec >= 0) {
    void engine.seek(args.seekSec);
  }
  renderUI(data.theme);
  applyYoutubeArgs(
    String(args.youtubeVideoId ?? ""),
    Boolean(args.hideYoutubeVideo),
    String(args.localVideoUrl ?? ""),
    Number(args.videoOffsetSec) || 0
  );
  setLoadProgress(0, stems.length, "Loading stems…");
  const liveBuffers: Record<string, (ctx: AudioContext, durationSec: number) => AudioBuffer> =
    {};
  if (metroConfig) {
    liveBuffers.metronome = (ctx, durationSec) => {
      const cfg: MetronomeConfig = {
        ...metroConfig!,
        durationSec: Math.max(metroConfig!.durationSec, durationSec, 0.05),
      };
      metroConfig = cfg;
      return buildMetronomeAudioBuffer(ctx, cfg, metroOptions);
    };
  }
  const { errors } = await engine.loadStems(
    stems,
    (loaded, total, msg) => {
      setLoadProgress(loaded, total, msg || `${loaded}/${total}`);
      scheduleFrameHeight();
    },
    liveBuffers
  );
  applyStateGains();
  applyDecodedPeaks();
  paintDecodedWaves();
  // Do not reportState(true) on load — that remounts the iframe via a parent
  // Streamlit rerun. Only publish when the user changes mute/solo/volume.
  // Do not renderUI again here: a second full rebuild flashes the picture and
  // transport after decode. Waveforms are painted into the rows already on screen.
  lastPublished = statePayload();
  finishLoadProgress(
    errors.length
      ? `Loaded with errors: ${errors.join("; ")}`
      : `Loaded ${stems.length} stem(s). Adjust levels while playing — playback will not restart.`
  );
  applyStateGains();
  restoreTransport();
  await applyInitialPractice(args);
  applyPracticeArgs(args);

  scheduleFrameHeight();
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
scheduleFrameHeight();

window.addEventListener("keydown", (e) => {
  if (e.repeat || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
  if (isTypingTarget(e.target)) return;
  const key = e.key.toLowerCase();
  if (key === "l") {
    e.preventDefault();
    const toggle = document.getElementById("loop-toggle") as HTMLInputElement | null;
    if (toggle) {
      toggle.checked = !toggle.checked;
      toggle.dispatchEvent(new Event("change"));
    }
  } else if (key === "c") {
    e.preventDefault();
    setCountInBars((countInBars + 1) % 3);
  }
});

window.addEventListener("keydown", (e) => {
  if (e.code !== "Space" && e.key !== " ") return;
  if (e.repeat || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
  if (isTypingTarget(e.target)) return;
  // Space on a group header folds it (disclosure buttons answer Enter and Space).
  if (e.target instanceof HTMLElement && e.target.closest(".group-toggle")) return;
  e.preventDefault();
  togglePlayPause();
});
