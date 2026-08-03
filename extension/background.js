// Minimal background service worker. Currently just a placeholder --
// Manifest V3 requires this file to exist since manifest.json references
// it, even though the popup flow doesn't need any background logic yet.
//
// Future extension point: a right-click context menu item ("Analyze
// selected text") could be added here using chrome.contextMenus, so
// analysis doesn't require opening the popup manually. Left out of this
// MVP to keep the first working version simple.

chrome.runtime.onInstalled.addListener(() => {
  console.log("Text Analysis Tester extension installed.");
});
