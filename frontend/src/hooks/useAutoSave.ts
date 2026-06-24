import { useEffect, useRef } from 'react';
import { useCanvasStore } from '../store/canvasStore';

const DEBOUNCE_MS = 2000;

export function useAutoSave() {
  const { canvasId, isDirty, saveCanvas } = useCanvasStore();
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!canvasId || !isDirty) return;

    if (timerRef.current) clearTimeout(timerRef.current);

    timerRef.current = setTimeout(() => {
      saveCanvas();
    }, DEBOUNCE_MS);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [canvasId, isDirty, saveCanvas]);
}
