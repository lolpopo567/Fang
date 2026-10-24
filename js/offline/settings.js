import { t } from '../i18n.js';
import * as pack from './pack.js';

const mb = (bytes) => Math.round(bytes / 1e6);

// The "Offline speech" row in Settings: download, progress and delete for the speech pack.
export function initSpeechPack() {
  const state = document.getElementById('pack-state');
  const get = document.getElementById('pack-download');
  const del = document.getElementById('pack-delete');
  const bar = document.getElementById('pack-progress');
  let installed = false;
  let busy = false;

  function show() {
    get.hidden = installed || busy;
    del.hidden = !installed || busy;
    bar.hidden = !busy;
    if (!busy) state.textContent = installed ? t('pack.installed') : t('pack.notInstalled').replace('{mb}', mb(pack.TOTAL_BYTES));
  }

  async function refresh() {
    installed = await pack.isInstalled();
    show();
  }

  if (!pack.supported()) {
    state.textContent = t('pack.unsupported');
    get.hidden = del.hidden = bar.hidden = true;
    return;
  }

  get.addEventListener('click', async () => {
    busy = true;
    show();
    try {
      await pack.install((f) => {
        bar.value = f;
        state.textContent = `${t('pack.downloading')} ${Math.floor(f * 100)}%`;
      });
      busy = false;
      await refresh();
    } catch (err) {
      busy = false;
      show();
      state.textContent = `${t('pack.failed')} (${err.message})`;
    }
    document.dispatchEvent(new Event('speechpackchange'));
  });

  del.addEventListener('click', async () => {
    if (!confirm(t('pack.confirmDelete'))) return;
    await pack.remove();
    await refresh();
    document.dispatchEvent(new Event('speechpackchange'));
  });

  document.addEventListener('langchange', show);
  refresh();
}
