#if UNITY_6000_3_OR_NEWER
using System;
using MCPForUnity.Editor.Services.RuntimeAnalysis;
using UnityEditor;
using UnityEditor.Toolbars;
using UnityEngine;
using UnityEngine.UIElements;

namespace MCPForUnity.Editor.Windows.Adjoint
{
    /// <summary>
    /// Registers Adjoint's main-toolbar elements (Unity 6.3+).
    ///   • "Adjoint/Run"  — middle dock, next to play. One-click start/stop play with Adjoint monitoring.
    ///   • "Adjoint/Menu" — left dock, alongside AI/Asset Store/Unity VCS. Dropdown hub for all Adjoint actions.
    /// File is excluded from compilation on Unity 6000.0–6000.2 via UNITY_6000_3_OR_NEWER.
    /// All actions remain reachable via the Adjoint menu and Cmd+Shift+{A,R,D} on every Unity version.
    /// Shipped as loose source in dist/ (NOT obfuscated) so the gate is evaluated against the user's actual Unity version.
    /// </summary>
    public static class AdjointToolbarButton
    {
        // Inlined from MCPForUnity.Editor.Constants.EditorPrefKeys.ToolbarAutoShown.
        // Inlined because this file is also distributed as a loose .cs in dist/Editor/Windows/Adjoint/
        // (see scripts/build-dist.sh) where it compiles into Assembly-CSharp-Editor and cannot reach
        // the internal EditorPrefKeys class inside Adjoint.Editor.dll.
        private const string ToolbarAutoShownKey = "Adjoint.ToolbarButton.AutoShown";
        private const string ToolbarMenuAutoShownKey = "Adjoint.ToolbarMenu.AutoShown";
        private const string AdjointLogoGuid = "7e28d1c61eaf24a3f9528d76a118740f";
        private const string AdjointLogoRelativePath = "Editor/Windows/Adjoint/Icons/adjoint-logo.png";
        private const string KnownPackagePath = "Packages/com.adjoint.editor";

        private static Texture2D _adjointIcon;
        
        /// <summary>
        /// Main "Run with Adjoint" button - positioned next to Play buttons.
        /// Click to start Play Mode with Adjoint monitoring.
        /// Click again while playing to stop.
        /// </summary>
        [MainToolbarElement("Adjoint/Run", defaultDockPosition = MainToolbarDockPosition.Middle)]
        public static MainToolbarElement RunWithAdjointButton()
        {
            LoadIcon();
            
            var isPlaying = EditorApplication.isPlaying;
            var service = AdjointRuntimeAnalysisService.Instance;
            var isAdjointActive = service.IsAdjointModeEnabled && isPlaying;
            
            // Show different icon based on state (no text, icon only to match play buttons)
            Texture2D icon;
            
            if (isAdjointActive)
            {
                // Currently running with Adjoint - show stop state
                icon = EditorGUIUtility.IconContent("d_PreMatQuad").image as Texture2D;
            }
            else
            {
                // Not running - show run state
                icon = _adjointIcon ?? EditorGUIUtility.IconContent("d_PlayButton").image as Texture2D;
            }
            
            var content = new MainToolbarContent(icon);
            var button = new MainToolbarButton(content, OnButtonClicked);
            
            // Style to match Unity's play buttons size
            MainToolbarElementStyler.StyleElement<VisualElement>("Adjoint/Run", element =>
            {
                // Match size of play buttons
                element.style.width = 35f;
                element.style.height = 20f;
                element.style.minWidth = 35f;
                element.style.maxWidth = 35f;
                element.style.paddingLeft = 0f;
                element.style.paddingRight = 0f;
                element.style.paddingTop = 0f;
                element.style.paddingBottom = 0f;
                element.style.marginLeft = 1f;
                element.style.marginRight = 0f;
                element.style.marginTop = 8f;
                
                // Match the dark grey background of play buttons
                element.style.backgroundColor = new Color(0.35f, 0.35f, 0.35f, 1f);
                element.style.borderTopLeftRadius = 4f;
                element.style.borderTopRightRadius = 4f;
                element.style.borderBottomLeftRadius = 4f;
                element.style.borderBottomRightRadius = 4f;
                
                // Size the icon
                var image = element.Q<Image>();
                if (image != null)
                {
                    image.style.width = 16f;
                    image.style.height = 16f;
                }
                
                if (isAdjointActive)
                {
                    // Active/recording state - green tint
                    element.style.backgroundColor = new Color(0.2f, 0.5f, 0.3f, 1f);
                }
            });
            
            return button;
        }
        
        private static void OnButtonClicked()
        {
            var service = AdjointRuntimeAnalysisService.Instance;

            if (EditorApplication.isPlaying)
            {
                // Currently playing - stop
                EditorApplication.isPlaying = false;
            }
            else
            {
                // Not playing - start with Adjoint
                service.RunWithAdjoint();
            }

            // Refresh button to update visual state
            MainToolbar.Refresh("Adjoint/Run");
        }

        /// <summary>
        /// "Adjoint" dropdown — left dock, alongside AI / Asset Store / Unity VCS.
        /// Acts as a hub: opens chat, settings, generation queue, etc.
        /// </summary>
        [MainToolbarElement("Adjoint/Menu", defaultDockPosition = MainToolbarDockPosition.Left)]
        public static MainToolbarElement AdjointMenuDropdown()
        {
            LoadIcon();
            var icon = _adjointIcon ?? EditorGUIUtility.IconContent("d_GUILayer Icon").image as Texture2D;
            var content = new MainToolbarContent("Adjoint", icon, "Adjoint AI assistant");
            return new MainToolbarDropdown(content, ShowAdjointMenu);
        }

        private static void ShowAdjointMenu(Rect activatorRect)
        {
            var menu = new GenericMenu();
            menu.AddItem(new GUIContent("Open Chat"),                false, AdjointChatWindow.ShowWindow);
            menu.AddItem(new GUIContent("Run Performance Analysis"), false, AdjointChatWindow.RunPerformanceAnalysis);
            menu.AddItem(new GUIContent("Debug Console Errors"),     false, AdjointConsoleDebugButton.DebugConsoleErrors);
            menu.AddSeparator("");
            menu.AddItem(new GUIContent("Generation Queue"),         false, AdjointEditorWindow.ShowWindow);
            menu.AddItem(new GUIContent("Clear Generation Queue"),   false, MCPForUnity.Editor.MenuItems.ClearGenerationQueue.Clear);
            menu.AddSeparator("");
            menu.AddItem(new GUIContent("Settings"),                 false, AdjointSettingsWindow.ShowWindow);
            menu.DropDown(activatorRect);
        }

        private static void LoadIcon()
        {
            if (_adjointIcon == null)
            {
                // Try to load custom Adjoint icon
                string basePath = Helpers.AssetPathUtility.GetMcpPackageRootPath();
                if (!string.IsNullOrEmpty(basePath))
                {
                    _adjointIcon = AssetDatabase.LoadAssetAtPath<Texture2D>(
                        $"{basePath}/{AdjointLogoRelativePath}");
                }

                // Loose-source dist installs can compile this file outside the package assembly.
                // The copied icon meta keeps this GUID stable even when package-root lookup misses.
                if (_adjointIcon == null)
                {
                    string guidPath = AssetDatabase.GUIDToAssetPath(AdjointLogoGuid);
                    if (!string.IsNullOrEmpty(guidPath))
                    {
                        _adjointIcon = AssetDatabase.LoadAssetAtPath<Texture2D>(guidPath);
                    }
                }

                if (_adjointIcon == null)
                {
                    _adjointIcon = AssetDatabase.LoadAssetAtPath<Texture2D>(
                        $"{KnownPackagePath}/{AdjointLogoRelativePath}");
                }
            }
        }
        
        /// <summary>
        /// Subscribe to play mode changes to refresh button state.
        /// Auto-show both toolbar elements on first install (each tracked under its own per-project pref key).
        /// </summary>
        [InitializeOnLoadMethod]
        private static void Initialize()
        {
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
            EditorApplication.delayCall += () => TryAutoShowToolbarElement("Adjoint/Run", ToolbarAutoShownKey);
            EditorApplication.delayCall += () => TryAutoShowToolbarElement("Adjoint/Menu", ToolbarMenuAutoShownKey);
            EditorApplication.delayCall += TryEnsureIconLoaded;
        }

        private static void OnPlayModeStateChanged(PlayModeStateChange state)
        {
            // Refresh button when play mode changes
            MainToolbar.Refresh("Adjoint/Run");
        }

        // On first install AssetDatabase may still be indexing the package when Unity
        // first invokes the toolbar factory, leaving _adjointIcon null. Poll briefly
        // and refresh the toolbar elements once the real icon resolves so the user does
        // not see the fallback icons until the next domain reload.
        private static double _iconLoadDeadline;
        private const double IconLoadTimeoutSeconds = 5.0;

        private static void TryEnsureIconLoaded()
        {
            // Reset the deadline on every exit path so this method is idempotent —
            // any future caller can re-arm a fresh polling window without being
            // blocked by a stale deadline from a prior load cycle.
            if (_adjointIcon != null)
            {
                _iconLoadDeadline = 0;
                return;
            }

            if (_iconLoadDeadline == 0)
                _iconLoadDeadline = EditorApplication.timeSinceStartup + IconLoadTimeoutSeconds;

            LoadIcon();
            if (_adjointIcon != null)
            {
                try
                {
                    MainToolbar.Refresh("Adjoint/Run");
                    MainToolbar.Refresh("Adjoint/Menu");
                }
                catch (Exception ex) { Debug.LogWarning($"[Adjoint] Toolbar refresh failed after icon load: {ex.Message}"); }
                _iconLoadDeadline = 0;
                return;
            }

            if (EditorApplication.timeSinceStartup < _iconLoadDeadline)
                EditorApplication.delayCall += TryEnsureIconLoaded;
            else
                _iconLoadDeadline = 0;
        }

        private static void TryAutoShowToolbarElement(string elementId, string baseKey)
        {
            var prefKey = $"{baseKey}.{Application.dataPath.GetHashCode()}";

            if (EditorPrefs.GetBool(prefKey, false))
                return;

            if (Application.isBatchMode)
                return;

            // Double delayCall to ensure toolbar is fully initialized
            EditorApplication.delayCall += () =>
            {
                if (ShowToolbarElement(elementId))
                {
                    EditorPrefs.SetBool(prefKey, true);
                    return;
                }

                // Retry once — toolbar initialization can be slow
                EditorApplication.delayCall += () =>
                {
                    if (ShowToolbarElement(elementId))
                    {
                        EditorPrefs.SetBool(prefKey, true);
                    }
                };
            };
        }

        private static bool ShowToolbarElement(string elementId)
        {
            try
            {
                var mainToolbarWindowType = typeof(EditorWindow).Assembly
                    .GetType("UnityEditor.MainToolbarWindow");
                if (mainToolbarWindowType == null)
                    return false;

                var toolbarWindows = UnityEngine.Resources.FindObjectsOfTypeAll(mainToolbarWindowType);
                if (toolbarWindows.Length == 0)
                    return false;

                var toolbarWindow = toolbarWindows[0] as EditorWindow;
                if (toolbarWindow == null)
                    return false;

                var overlayCanvas = toolbarWindow.overlayCanvas;
                if (overlayCanvas == null)
                    return false;

                foreach (var overlay in overlayCanvas.overlays)
                {
                    if (overlay.id == elementId)
                    {
                        if (!overlay.displayed)
                            overlay.displayed = true;
                        return true;
                    }
                }

                return false;
            }
            catch (Exception)
            {
                return false;
            }
        }
    }
    
    /// <summary>
    /// VisualElement extension methods for finding elements by name or tooltip.
    /// </summary>
    public static class VisualElementExtensions
    {
        public static VisualElement FindElementByName(this VisualElement root, string name)
        {
            if (root.name == name) return root;
            
            foreach (var child in root.Children())
            {
                var found = child.FindElementByName(name);
                if (found != null) return found;
            }
            
            return null;
        }
        
        public static VisualElement FindElementByTooltip(this VisualElement root, string tooltip)
        {
            if (root.tooltip == tooltip) return root;
            
            foreach (var child in root.Children())
            {
                var found = child.FindElementByTooltip(tooltip);
                if (found != null) return found;
            }
            
            return null;
        }
    }
    
    /// <summary>
    /// Helper class for styling toolbar elements.
    /// Based on Unity 6.3 toolbar styling patterns.
    /// </summary>
    public static class MainToolbarElementStyler
    {
        public static void StyleElement<T>(string elementName, System.Action<T> styleAction) where T : VisualElement
        {
            EditorApplication.delayCall += () =>
            {
                ApplyStyle(elementName, element =>
                {
                    T targetElement = null;
                    
                    if (element is T typedElement)
                    {
                        targetElement = typedElement;
                    }
                    else
                    {
                        targetElement = element.Query<T>().First();
                    }
                    
                    if (targetElement != null)
                    {
                        styleAction(targetElement);
                    }
                });
            };
        }
        
        private static void ApplyStyle(string elementName, System.Action<VisualElement> styleCallback)
        {
            var element = FindElementByName(elementName);
            if (element != null)
            {
                styleCallback(element);
            }
        }
        
        private static VisualElement FindElementByName(string name)
        {
            var nameWithoutSpace = name.Replace(" ", "");
            var windows = UnityEngine.Resources.FindObjectsOfTypeAll<EditorWindow>();
            
            foreach (var window in windows)
            {
                var root = window.rootVisualElement;
                if (root == null) continue;
                
                VisualElement element;
                if ((element = root.FindElementByName(name)) != null) return element;
                if ((element = root.FindElementByName(nameWithoutSpace)) != null) return element;
                if ((element = root.FindElementByTooltip(name)) != null) return element;
            }

            return null;
        }
    }
}
#endif
