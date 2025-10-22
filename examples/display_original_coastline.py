#!/usr/bin/env python3
"""
Display the full original Singapore coastline data with colored segments.

This script loads the original `singapore.geojson` file, which contains
multiple disconnected coastline segments, and plots each segment with a
unique color using a continuous colormap. A colorbar is added as a legend.
"""

from pathlib import Path
import geopandas as gpd
import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.widgets import Button, RectangleSelector, TextBox
from shapely.geometry import box, Point


def display_original_coastline(input_path: Path):
    """
    Loads and displays all coastline segments from a GeoJSON file with unique colors.

    Args:
        input_path: Path to the GeoJSON file.
    """
    if not input_path.exists():
        print(f"Error: Data not found at {input_path}")
        return

    print(f"Loading data from {input_path}...")
    gdf = gpd.read_file(input_path)
    gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)
    gdf = gdf.reset_index(drop=True)
    lengths = gdf.geometry.length
    gdf["_length"] = lengths

    num_segments = len(gdf)
    print(f"Found {num_segments} coastline segments.")
    print(f"GeoDataFrame shape: {gdf.shape}")
    if num_segments > 0:
        print(f"Sample geometry type: {gdf.geometry.iloc[0].geom_type}")
        print(f"Sample bounds: {gdf.geometry.iloc[0].bounds}")
        print(f"Total bounds: {gdf.total_bounds}")

    if num_segments == 0:
        print("No geometries to display.")
        return

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_axes([0.05, 0.10, 0.68, 0.85])  # main plotting axis
    ax_info = fig.add_axes([0.76, 0.35, 0.22, 0.55])  # info panel
    ax_info.axis("off")

    # Right-side controls (stacked under the info panel)
    btn_box_ax = fig.add_axes([0.76, 0.28, 0.22, 0.045])
    btn_dedupe_ax = fig.add_axes([0.76, 0.225, 0.22, 0.045])
    btn_remove_small_ax = fig.add_axes([0.76, 0.17, 0.12, 0.045])
    tb_len_ax = fig.add_axes([0.89, 0.17, 0.09, 0.045])
    btn_labels_ax = fig.add_axes([0.76, 0.115, 0.12, 0.045])
    btn_hide_ax = fig.add_axes([0.89, 0.115, 0.09, 0.045])
    btn_reset_ax = fig.add_axes([0.76, 0.06, 0.22, 0.045])
    btn_export_ax = fig.add_axes([0.76, 0.01, 0.22, 0.045])

    btn_box = Button(btn_box_ax, "Box select (b)")
    btn_dedupe = Button(btn_dedupe_ax, "Auto-dedupe (d)")
    btn_remove_small = Button(btn_remove_small_ax, "Remove small (s)")
    tb_len = TextBox(tb_len_ax, "len thresh", initial="5%")
    btn_labels = Button(btn_labels_ax, "Toggle labels (l)")
    btn_hide = Button(btn_hide_ax, "Hide removed (h)")
    btn_reset = Button(btn_reset_ax, "Reset (r)")
    btn_export = Button(btn_export_ax, "Export (e)")

    # Choose colormap: discrete for small N, continuous for large N
    if num_segments <= 20:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % cmap.N) for i in range(num_segments)]
        cmap_for_cb = mcolors.ListedColormap(colors)
        norm = mcolors.Normalize(vmin=0, vmax=num_segments - 1)
    else:
        cmap = plt.get_cmap("viridis")
        norm = mcolors.Normalize(vmin=0, vmax=num_segments - 1)
        colors = [cmap(norm(i)) for i in range(num_segments)]
        cmap_for_cb = cmap

    # Helper: compute a representative midpoint for a segment (for label placement)
    def _segment_midpoint_coords(geom):
        # Return a representative x,y coordinate for labeling. Avoid broad exception
        # handling by checking attributes explicitly.
        if geom is None or getattr(geom, "is_empty", True):
            return None, None

        gtype = getattr(geom, "geom_type", None)
        if gtype == "LineString":
            xy = getattr(geom, "xy", None)
            if xy:
                x, y = xy
                if len(x) > 0:
                    return x[len(x) // 2], y[len(y) // 2]
        elif gtype == "MultiLineString":
            geoms = getattr(geom, "geoms", None)
            if geoms:
                longest = max(geoms, key=lambda g: g.length)
                xy = getattr(longest, "xy", None)
                if xy:
                    x, y = xy
                    if len(x) > 0:
                        return x[len(x) // 2], y[len(y) // 2]

        # Fallbacks
        if hasattr(geom, "representative_point"):
            pt = geom.representative_point()
            return pt.x, pt.y

        return None, None

    # Data structures to track plotted lines and annotations
    lines_by_segment = {}  # idx -> [Line2D, ...] for multi-segment geoms
    annotations = {}  # idx -> Text
    original_colors = {}
    removed = set()
    plotted_count = 0  # Counter for debugging

    # Decide whether to show labels by default (avoid clutter for very large datasets)
    show_labels = num_segments <= 200

    # Plot each segment as one or more Line2D objects and attach a picker
    for i, geom in enumerate(gdf.geometry):
        seg_lines = []
        color = colors[i]
        original_colors[i] = color
        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == "LineString":
            x, y = geom.xy
            (line,) = ax.plot(x, y, color=color, linewidth=1, picker=5)
            line.set_gid(str(i))
            seg_lines.append(line)
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                x, y = part.xy
                (line,) = ax.plot(x, y, color=color, linewidth=1, picker=5)
                line.set_gid(str(i))
                seg_lines.append(line)
        elif geom.geom_type == "Polygon":
            x, y = geom.exterior.xy
            (line,) = ax.plot(x, y, color=color, linewidth=1, picker=5)
            line.set_gid(str(i))
            seg_lines.append(line)
        elif geom.geom_type == "MultiPolygon":
            for part in geom.geoms:
                x, y = part.exterior.xy
                (line,) = ax.plot(x, y, color=color, linewidth=1, picker=5)
                line.set_gid(str(i))
                seg_lines.append(line)
        else:
            # ignore unexpected geometry types
            continue

        if seg_lines:
            plotted_count += 1
        lines_by_segment[i] = seg_lines

        # Add (optional) small numeric label for each segment to make identification easier
        mx, my = _segment_midpoint_coords(geom)
        if mx is not None and my is not None:
            if show_labels:
                txt = ax.text(
                    mx,
                    my,
                    str(i),
                    fontsize=6,
                    color=color,
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.1",
                        facecolor="white",
                        alpha=0.7,
                        linewidth=0,
                    ),
                )
            else:
                txt = ax.text(
                    mx,
                    my,
                    str(i),
                    fontsize=6,
                    color=color,
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.1",
                        facecolor="white",
                        alpha=0.0,
                        linewidth=0,
                    ),
                    visible=False,
                )
            annotations[i] = txt

    # Ensure the axis bounds include all plotted geometries so the coastline is visible
    try:
        minx, miny, maxx, maxy = gdf.total_bounds
        # Add a small padding relative to the larger dimension
        dx = max(maxx - minx, 1e-12)
        dy = max(maxy - miny, 1e-12)
        pad = max(dx, dy) * 0.02
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
    except Exception:
        # Fallback to autoscale which should still show lines if present
        ax.relim()
        ax.autoscale_view()

    # Debugging: Print axis limits and plotted count
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    print(f"Axis limits set to x: {xlim}, y: {ylim}")
    print(f"Number of segments plotted: {plotted_count}")

    ax.set_title("Original Singapore Coastline - Interactive segment selection")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")

    # Add colorbar as legend (use a small set of ticks for large N)
    sm = plt.cm.ScalarMappable(cmap=cmap_for_cb, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    if num_segments <= 20:
        cbar.set_ticks(np.arange(num_segments))
        cbar.set_ticklabels([str(i) for i in range(num_segments)])
    else:
        ticks = np.linspace(0, num_segments - 1, num=min(6, num_segments)).astype(int)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(int(t)) for t in ticks])
    cbar.set_label("Segment Index (0 to {})".format(num_segments - 1))

    # Info panel: dynamic status and brief instructions
    info_text = ax_info.text(0, 1, "", va="top", fontsize=9, wrap=True)

    def _format_removed_list(max_items=200):
        removed_list = sorted(removed)
        if not removed_list:
            return "None"
        if len(removed_list) <= max_items:
            return ", ".join(map(str, removed_list))
        else:
            head = ", ".join(map(str, removed_list[:max_items]))
            return f"{head}, ... (+{len(removed_list)-max_items} more)"

    def update_info_panel(message: str = ""):
        remaining_count = num_segments - len(removed)
        info_lines = [
            "Instructions:",
            " - Click a segment to toggle remove/restore",
            " - Buttons: Reset (r), Export (e), Toggle labels (l)",
            "",
            f"Total segments: {num_segments}",
            f"Remaining: {remaining_count}",
            f"Removed: {len(removed)}",
            "",
            "Removed indices:",
            _format_removed_list(200),
            "",
            message,
        ]
        info_text.set_text("\n".join(info_lines))
        fig.canvas.draw_idle()

    # Small UI state
    hide_removed = False

    # Helper to apply visual state for a single segment index
    def _apply_removed_state(idx, remove=True):
        if idx not in lines_by_segment:
            return
        if remove:
            if idx in removed:
                return
            removed.add(idx)
            for ln in lines_by_segment.get(idx, []):
                if hide_removed:
                    ln.set_visible(False)
                else:
                    ln.set_color("lightgray")
                    ln.set_alpha(0.25)
                    ln.set_linewidth(0.5)
            if idx in annotations:
                if hide_removed:
                    annotations[idx].set_visible(False)
                else:
                    annotations[idx].set_color("lightgray")
                    annotations[idx].set_alpha(0.25)
                    annotations[idx].set_visible(True)
        else:
            if idx not in removed:
                return
            removed.discard(idx)
            for ln in lines_by_segment.get(idx, []):
                ln.set_color(original_colors.get(idx, "black"))
                ln.set_alpha(1.0)
                ln.set_linewidth(1.0)
                ln.set_visible(True)
            if idx in annotations:
                annotations[idx].set_color(original_colors.get(idx, "black"))
                annotations[idx].set_alpha(1.0)
                annotations[idx].set_visible(show_labels)

    def toggle_segment(idx):
        _apply_removed_state(idx, remove=(idx not in removed))

    def toggle_multiple(idxs):
        count = 0
        for idx in idxs:
            if 0 <= idx < num_segments:
                toggle_segment(idx)
                count += 1
        update_info_panel(f"Toggled {count} segments via selection.")

    def _px_to_data_radius(px=8):
        """Estimate a data-coordinate radius corresponding to px screen pixels.

        This uses the axis data limits and axis pixel size to avoid using
        transforms that may not always be available in headless backends.
        """
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        bbox = ax.get_window_extent()
        width_px = bbox.width if bbox.width > 0 else 1.0
        height_px = bbox.height if bbox.height > 0 else 1.0
        dx_per_px = abs(x1 - x0) / width_px
        dy_per_px = abs(y1 - y0) / height_px
        return max(dx_per_px * px, dy_per_px * px)

    def _find_candidates_near(xdata, ydata, px_tol=8):
        r = _px_to_data_radius(px_tol)
        bbox = (xdata - r, ydata - r, xdata + r, ydata + r)
        candidates = list(gdf.sindex.intersection(bbox))
        # filter by actual distance
        pt = Point(xdata, ydata)
        close = []
        for idx in candidates:
            if not (0 <= idx < num_segments):
                continue
            geom = gdf.geometry.iloc[idx]
            if geom is None or getattr(geom, "is_empty", True):
                continue
            d = geom.distance(pt)
            if d <= r * 2:
                close.append((idx, d))
        close.sort(key=lambda x: x[1])
        return [c[0] for c in close]

    # Event handlers
    def on_pick(event):
        artist = event.artist
        gid = artist.get_gid() if hasattr(artist, "get_gid") else None
        if not isinstance(gid, str) or not gid.isdigit():
            return
        idx = int(gid)

        if idx in removed:
            # restore
            removed.remove(idx)
            for ln in lines_by_segment.get(idx, []):
                ln.set_color(original_colors[idx])
                ln.set_alpha(1.0)
                ln.set_linewidth(1.0)
            if idx in annotations:
                annotations[idx].set_color(original_colors[idx])
                annotations[idx].set_alpha(1.0)
                annotations[idx].set_visible(show_labels)
        else:
            # mark removed (dim + grey)
            removed.add(idx)
            for ln in lines_by_segment.get(idx, []):
                ln.set_color("lightgray")
                ln.set_alpha(0.25)
                ln.set_linewidth(0.5)
            if idx in annotations:
                annotations[idx].set_color("lightgray")
                annotations[idx].set_alpha(0.25)
                annotations[idx].set_visible(True)

        # prepare a short summary about the toggled segment (safe indexing)
        if 0 <= idx < num_segments:
            seg_geom = gdf.geometry.iloc[idx]
            parts = (
                1
                if getattr(seg_geom, "geom_type", "") == "LineString"
                else len(getattr(seg_geom, "geoms", []))
            )
            seg_length = float(getattr(seg_geom, "length", 0.0))
            action = "Removed" if idx in removed else "Restored"
            message = f"Last action: {action} segment {idx} — length={seg_length:.3f}, parts={parts}"
        else:
            message = ""

        update_info_panel(message)

    def _reset(_event=None):
        removed.clear()
        for idx, seg_lines in lines_by_segment.items():
            for ln in seg_lines:
                ln.set_color(original_colors[idx])
                ln.set_alpha(1.0)
                ln.set_linewidth(1.0)
        # restore annotation visibility to previous label state
        for idx, txt in annotations.items():
            txt.set_color(original_colors[idx])
            txt.set_alpha(1.0)
            txt.set_visible(show_labels)
        update_info_panel("Reset all selections.")

    def _export(_event=None):
        remaining = sorted(set(range(num_segments)) - removed)
        out_path = input_path.parent / f"{input_path.stem}_remaining_indices.txt"
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                for idx in remaining:
                    fh.write(f"{idx}\n")
            update_info_panel(f"Exported {len(remaining)} indices to {out_path}")
            print(f"Exported remaining indices to {out_path}")
        except OSError as exc:
            update_info_panel(f"Failed to export: {exc}")
            print(f"Failed to export remaining indices: {exc}")

    def _toggle_labels(_event=None):
        nonlocal show_labels
        show_labels = not show_labels
        for idx, txt in annotations.items():
            txt.set_visible(show_labels and (idx not in removed))
        update_info_panel(f"Labels {'shown' if show_labels else 'hidden'}.")

    def on_key(event):
        if event.key == "r":
            _reset()
        elif event.key == "e":
            _export()
        elif event.key == "l":
            _toggle_labels()
        elif event.key == "b":
            _toggle_box_select()
        elif event.key == "d":
            _auto_dedupe()
        elif event.key == "s":
            _remove_small()
        elif event.key == "h":
            _toggle_hide_removed()

    # Wire control callbacks
    btn_reset.on_clicked(_reset)
    btn_export.on_clicked(_export)
    btn_labels.on_clicked(_toggle_labels)

    # Rectangle selector (inactive by default)
    def _on_box_select(eclick, erelease):
        # eclick/erelease are mouse events with data coordinates
        if (
            eclick.xdata is None
            or eclick.ydata is None
            or erelease.xdata is None
            or erelease.ydata is None
        ):
            update_info_panel("Box selection cancelled.")
            return
        minx = min(eclick.xdata, erelease.xdata)
        maxx = max(eclick.xdata, erelease.xdata)
        miny = min(eclick.ydata, erelease.ydata)
        maxy = max(eclick.ydata, erelease.ydata)
        sel_box = box(minx, miny, maxx, maxy)
        candidates = list(gdf.sindex.intersection(sel_box.bounds))
        selected = [i for i in candidates if gdf.geometry.iloc[i].intersects(sel_box)]
        if not selected:
            update_info_panel("No segments in selection box.")
            return
        toggle_multiple(selected)

    rs = RectangleSelector(ax, onselect=_on_box_select)
    rs.set_active(False)

    def _toggle_box_select(_event=None):
        active = getattr(rs, "active", False)
        rs.set_active(not active)
        btn_box.label.set_text("Box select (b) ON" if not active else "Box select (b)")
        update_info_panel(
            "Box select {}".format("activated" if not active else "deactivated")
        )

    btn_box.on_clicked(_toggle_box_select)

    # Remove small segments using the threshold text box (supports absolute or percentage, e.g. '5%')
    def _parse_threshold_text(txt: str, reference_value: float):
        t = (txt or "").strip()
        if not t:
            return None
        if t.endswith("%"):
            try:
                pct = float(t.strip("%")) / 100.0
                return float(reference_value) * pct
            except ValueError:
                return None
        try:
            return float(t)
        except ValueError:
            return None

    def _remove_small(_event=None):
        txt = tb_len.text
        ref = float(gdf["_length"].max() if len(gdf) else 1.0)
        thr = _parse_threshold_text(txt, ref)
        if thr is None:
            update_info_panel("Invalid length threshold: '{}'".format(txt))
            return
        to_remove = [
            i for i, l in enumerate(gdf["_length"]) if l <= thr and i not in removed
        ]
        for idx in to_remove:
            _apply_removed_state(idx, True)
        update_info_panel(f"Removed {len(to_remove)} segments with length <= {thr:.6f}")

    btn_remove_small.on_clicked(_remove_small)

    # Auto dedupe overlapping segments: remove the shorter of any two that overlap
    # by at least the given ratio of the shorter length (default 0.85)
    tb_overlap_ax = fig.add_axes([0.89, 0.225, 0.09, 0.045])
    tb_overlap = TextBox(tb_overlap_ax, "ov thresh", initial="85%")

    def _parse_ratio(txt: str):
        if not txt:
            return None
        t = txt.strip()
        if t.endswith("%"):
            try:
                return float(t.strip("%")) / 100.0
            except ValueError:
                return None
        try:
            v = float(t)
            if v > 1.0:
                return v
            return v
        except ValueError:
            return None

    def _auto_dedupe(_event=None):
        txt = tb_overlap.text
        ratio = _parse_ratio(txt)
        if ratio is None:
            update_info_panel(f"Invalid overlap threshold: {txt}")
            return
        to_remove = set()
        for i in range(num_segments):
            if i in removed or i in to_remove:
                continue
            geom_i = gdf.geometry.iloc[i]
            if geom_i is None or getattr(geom_i, "is_empty", True):
                continue
            candidates = list(gdf.sindex.intersection(geom_i.bounds))
            for j in candidates:
                if j <= i or j in removed or j in to_remove:
                    continue
                geom_j = gdf.geometry.iloc[j]
                if geom_j is None or getattr(geom_j, "is_empty", True):
                    continue
                if not geom_i.intersects(geom_j):
                    continue
                inter = geom_i.intersection(geom_j)
                inter_len = float(getattr(inter, "length", 0.0))
                min_len = min(
                    float(getattr(geom_i, "length", 0.0)),
                    float(getattr(geom_j, "length", 0.0)),
                )
                if min_len <= 0:
                    continue
                if inter_len / min_len >= ratio:
                    # remove the shorter
                    if geom_i.length < geom_j.length:
                        to_remove.add(i)
                    else:
                        to_remove.add(j)
        for idx in sorted(to_remove):
            _apply_removed_state(idx, True)
        update_info_panel(
            f"Auto-dedupe removed {len(to_remove)} segments (threshold={ratio})"
        )

    btn_dedupe.on_clicked(_auto_dedupe)

    # Hide/Show removed
    def _toggle_hide_removed(_event=None):
        nonlocal hide_removed
        hide_removed = not hide_removed
        for idx in removed:
            for ln in lines_by_segment.get(idx, []):
                ln.set_visible(not hide_removed)
            if idx in annotations:
                annotations[idx].set_visible(not hide_removed)
        btn_hide.label.set_text(
            "Show removed (h)" if hide_removed else "Hide removed (h)"
        )
        update_info_panel(
            "Removed segments {}".format("hidden" if hide_removed else "visible")
        )

    btn_hide.on_clicked(_toggle_hide_removed)

    # Nearest-click fallback and multi-select support for overlapping segments
    def on_click(event):
        # Only act on left clicks inside the main axis
        if event.inaxes is not ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        # If any artist contains the event, let pick_event handle normal toggling
        # (pick_event will also handle shift-click multi-selection).
        for seg_lines in lines_by_segment.values():
            for artist in seg_lines:
                contains, _ = artist.contains(event)
                if contains:
                    return
        # Fallback: find nearest segments
        idxs = _find_candidates_near(event.xdata, event.ydata, px_tol=8)
        if not idxs:
            update_info_panel("No nearby segment to toggle.")
            return
        if getattr(event, "key", None) and "shift" in event.key:
            toggle_multiple(idxs)
        else:
            toggle_segment(idxs[0])

    fig.canvas.mpl_connect("button_press_event", on_click)

    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("key_press_event", on_key)

    # initial info
    update_info_panel()

    # small instruction on the map
    ax.text(
        0.01,
        -0.05,
        "Click segments to toggle removal. Use buttons or keys: r=reset, e=export, l=toggle labels.",
        transform=ax.transAxes,
        fontsize=9,
    )

    # Explicitly reference UI objects and helper functions so static analysis
    # tools that scan for "unused" names see they are intentionally retained
    # (they are used by Matplotlib as callbacks).
    _unused = (
        btn_box,
        btn_dedupe,
        btn_remove_small,
        tb_len,
        tb_overlap,
        btn_labels,
        btn_hide,
        toggle_multiple,
        RectangleSelector,
        box,
    )
    del _unused

    output_path = input_path.parent / f"{input_path.stem}_display.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Coastline plot saved to {output_path}")
    print("If the plot window did not appear, check the saved image file.")

    plt.show()


def main():
    input_file = Path("examples/data/singapore_2018.geojson")
    display_original_coastline(input_file)


if __name__ == "__main__":
    main()
