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
    /// Adds "Run with Adjoint" button to the Unity main toolbar.
    /// Uses Unity 6.3+ MainToolbar API.
    /// File is excluded from compilation on Unity 6000.0–6000.2 via UNITY_6000_3_OR_NEWER.
    /// The same action remains available via the Adjoint menu and Cmd+Shift+R on all 6.x versions.
    /// Shipped as loose source in dist/ (NOT obfuscated) so the gate is evaluated against the user's actual Unity version.
    /// Single button: Click to start Play Mode with Adjoint monitoring, click again to stop.
    /// </summary>
    public static class AdjointToolbarButton
    {
        // Inlined from MCPForUnity.Editor.Constants.EditorPrefKeys.ToolbarAutoShown.
        // Inlined because this file is also distributed as a loose .cs in dist/Editor/Windows/Adjoint/
        // (see scripts/build-dist.sh) where it compiles into Assembly-CSharp-Editor and cannot reach
        // the internal EditorPrefKeys class inside Adjoint.Editor.dll.
        private const string ToolbarAutoShownKey = "Adjoint.ToolbarButton.AutoShown";

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
                icon = _adjointIcon;
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
        
        private static void LoadIcon()
        {
            if (_adjointIcon == null)
            {
                // Try to load custom Adjoint icon
                string basePath = Helpers.AssetPathUtility.GetMcpPackageRootPath();
                _adjointIcon = AssetDatabase.LoadAssetAtPath<Texture2D>(
                    $"{basePath}/Editor/Windows/Adjoint/Icons/adjoint-logo.png");
                
                // Fallback to built-in icon if not found
                if (_adjointIcon == null)
                {
                    _adjointIcon = EditorGUIUtility.IconContent("d_PlayButton").image as Texture2D;
                }
            }
        }
        
        /// <summary>
        /// Subscribe to play mode changes to refresh button state.
        /// Auto-show toolbar button on first install.
        /// </summary>
        [InitializeOnLoadMethod]
        private static void Initialize()
        {
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
            EditorApplication.delayCall += TryAutoShowToolbarButton;
        }

        private static void OnPlayModeStateChanged(PlayModeStateChange state)
        {
            // Refresh button when play mode changes
            MainToolbar.Refresh("Adjoint/Run");
        }

        private static string ProjectSpecificPrefKey =>
            $"{ToolbarAutoShownKey}.{Application.dataPath.GetHashCode()}";

        private static void TryAutoShowToolbarButton()
        {
            if (EditorPrefs.GetBool(ProjectSpecificPrefKey, false))
                return;

            if (Application.isBatchMode)
                return;

            // Double delayCall to ensure toolbar is fully initialized
            EditorApplication.delayCall += () =>
            {
                if (ShowToolbarElement("Adjoint/Run"))
                {
                    EditorPrefs.SetBool(ProjectSpecificPrefKey, true);
                    return;
                }

                // Retry once — toolbar initialization can be slow
                EditorApplication.delayCall += () =>
                {
                    if (ShowToolbarElement("Adjoint/Run"))
                    {
                        EditorPrefs.SetBool(ProjectSpecificPrefKey, true);
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
