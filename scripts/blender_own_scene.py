"""Decide which objects a live Blender script may delete.

A long-lived Blender holds the user's scenes. ``bpy.data.objects.remove`` deletes
an object from every scene, so a script may delete only objects that live in its
own scene and in no other.
"""

from __future__ import annotations


def objects_exclusive_to_scene(scene_names, other_names) -> list[str]:
    """Names this call may delete: in its scene, and not in any other scene."""
    others = set(other_names)
    seen: set[str] = set()
    exclusive = []
    for name in scene_names:
        if name in others or name in seen:
            continue
        seen.add(name)
        exclusive.append(name)
    return exclusive


def own_scene_body(scene_name: str) -> str:
    """Blender statements that switch to one private scene and clear only that.

    ``bpy`` is already imported, and ``objects_exclusive_to_scene`` is already
    imported, when this text runs. The returned text has no format braces.
    """
    return f"""
_own_prev = bpy.context.window.scene
_own_scene = bpy.data.scenes.get({scene_name!r}) or bpy.data.scenes.new({scene_name!r})
bpy.context.window.scene = _own_scene
if bpy.context.scene.world is None:
    bpy.context.scene.world = bpy.data.worlds.new("IMG3D world")
_own_names = [obj.name for obj in _own_scene.objects]
_other_names = [obj.name for scene in bpy.data.scenes if scene != _own_scene for obj in scene.objects]
for _own_name in objects_exclusive_to_scene(_own_names, _other_names):
    bpy.data.objects.remove(bpy.data.objects[_own_name], do_unlink=True)
_own_cols = [_own_scene.collection]
_own_stack = list(_own_scene.collection.children)
while _own_stack:
    _own_col = _own_stack.pop()
    _own_cols.append(_own_col)
    _own_stack.extend(list(_own_col.children))
for _own_obj in list(_own_scene.objects):
    for _own_col in list(_own_obj.users_collection):
        if _own_col in _own_cols:
            _own_col.objects.unlink(_own_obj)
"""
