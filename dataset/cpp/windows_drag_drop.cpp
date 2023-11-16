// windows_drag_drop.cpp
/*
  neogfx C++ App/Game Engine
  Copyright (c) 2020 Leigh Johnston
  
  This program is free software: you can redistribute it and / or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <neogfx/neogfx.hpp>

#include <WinUser.h>
#include <comdef.h>
#include <shellapi.h>

#include <neolib/file/file.hpp>

#include <neogfx/hid/i_native_surface.hpp>
#include <neogfx/gui/window/i_native_window.hpp>
#include "windows_drag_drop.hpp"

namespace neogfx
{
    namespace native::windows
    {
        drag_drop::drag_drop() :
            base_type::drag_drop{}
        {
        }

        void drag_drop::register_target(i_drag_drop_target& aTarget)
        {
            base_type::register_target(aTarget);
            if (aTarget.is_widget() && aTarget.as_widget().root().has_native_surface())
            {
                HWND hwndTarget = static_cast<HWND>(aTarget.as_widget().root().native_window().native_handle());
                if (++iNativeDragDropTargets[hwndTarget].second == 1)
                {
                    iNativeDragDropTargets[hwndTarget].first.emplace(aTarget.as_widget().root().native_window());
                    auto result = ::RegisterDragDrop(hwndTarget, this);
                    if (result != S_OK)
                        throw failed_drag_drop_registration(neolib::utf16_to_utf8(reinterpret_cast<const char16_t*>(_com_error(result).ErrorMessage())));
                }
            }
        }

        void drag_drop::unregister_target(i_drag_drop_target& aTarget)
        {
            if (aTarget.is_widget() && aTarget.as_widget().root().has_native_surface())
            {
                HWND hwndTarget = static_cast<HWND>(aTarget.as_widget().root().native_window().native_handle());
                auto existing = iNativeDragDropTargets.find(hwndTarget);
                if (existing != iNativeDragDropTargets.end())
                {
                    if (!existing->second.first->is_destroyed() && --existing->second.second == 0)
                    {
                        iNativeDragDropTargets.erase(existing);
                        auto result = ::RevokeDragDrop(hwndTarget);
                        if (result != S_OK)
                            throw failed_drag_drop_unregistration(neolib::utf16_to_utf8(reinterpret_cast<const char16_t*>(_com_error(result).ErrorMessage())));
                    }
                }
                else
                {
                    for (auto i = iNativeDragDropTargets.begin(); i != iNativeDragDropTargets.end();)
                    {
                        if (i->second.first->is_destroyed())
                            i = iNativeDragDropTargets.erase(i);
                        else
                            ++i;
                    }
                }
            }
            base_type::unregister_target(aTarget);
        }

        namespace detail
        {
            class internal_drag_drop_source : public drag_drop_source<>
            {
            protected:
                bool is_drag_drop_object(point const& aPosition) const override
                {
                    return false;
                }
                i_drag_drop_object const* drag_drop_object(point const& aPosition) override
                {
                    return nullptr;
                }
            };
        }

        HRESULT drag_drop::DragEnter(IDataObject* pDataObj, DWORD grfKeyState, POINTL pt, DWORD* pdwEffect)
        {
            *pdwEffect = DROPEFFECT_NONE;

            FORMATETC fmtetc = { CF_HDROP, {}, DVASPECT_CONTENT, {}, TYMED_HGLOBAL };
            STGMEDIUM stgmed;

            if (pDataObj->GetData(&fmtetc, &stgmed) == S_OK)
            {
                HDROP hdrop = (HDROP)GlobalLock(stgmed.hGlobal);
                if (hdrop != NULL)
                {
                    std::vector<string> files;
                    auto fileCount = DragQueryFile(hdrop, (UINT)-1, NULL, 0u);
                    for (UINT i = 0u; i < fileCount; ++i)
                    {
                        std::wstring path(MAX_PATH + 1u, L'\0');
                        DragQueryFile(hdrop, i, &path[0u], (UINT)MAX_PATH + 1u);
                        path.resize(std::wcslen(path.c_str()));
                        if (!path.empty())
                            files.push_back(neolib::convert_path(path));
                    }
                    if (!files.empty())
                    {
                        iActiveInternalDragDropSource = std::make_unique<detail::internal_drag_drop_source>();
                        iActiveInternalDragDropObject = std::make_unique<drag_drop_file_list>(*iActiveInternalDragDropSource, files);
                        iActiveInternalDragDropSource->start_drag_drop(*iActiveInternalDragDropObject);
                        auto target = find_target(*iActiveInternalDragDropObject, basic_point<LONG>{pt.x, pt.y});
                        if (target != nullptr)
                            switch (target->accepted_as(*iActiveInternalDragDropObject))
                            {
                            case drop_operation::Copy:
                                *pdwEffect = DROPEFFECT_COPY;
                                break;
                            case drop_operation::Move:
                                *pdwEffect = DROPEFFECT_MOVE;
                                break;
                            case drop_operation::Link:
                                *pdwEffect = DROPEFFECT_LINK;
                                break;
                            }
                    }
                }
                ::ReleaseStgMedium(&stgmed);
            }

            return S_OK;
        }

        HRESULT drag_drop::DragOver(DWORD grfKeyState, POINTL pt, DWORD* pdwEffect)
        {
            *pdwEffect = DROPEFFECT_NONE;
            if (iActiveInternalDragDropObject)
            {
                auto target = find_target(*iActiveInternalDragDropObject, basic_point<LONG>{pt.x, pt.y});
                if (target != nullptr)
                    switch (target->accepted_as(*iActiveInternalDragDropObject))
                    {
                    case drop_operation::Copy:
                        *pdwEffect = DROPEFFECT_COPY;
                        break;
                    case drop_operation::Move:
                        *pdwEffect = DROPEFFECT_MOVE;
                        break;
                    case drop_operation::Link:
                        *pdwEffect = DROPEFFECT_LINK;
                        break;
                    }
            }
            return S_OK;
        }

        HRESULT drag_drop::DragLeave()
        {
            if (iActiveInternalDragDropObject)
            {
                iActiveInternalDragDropSource->cancel_drag_drop();
                iActiveInternalDragDropSource = nullptr;
                iActiveInternalDragDropObject = nullptr;
            }
            return S_OK;
        }

        HRESULT drag_drop::Drop(IDataObject* pDataObj, DWORD grfKeyState, POINTL pt, DWORD* pdwEffect)
        {
            if (iActiveInternalDragDropObject)
            {
                auto target = find_target(*iActiveInternalDragDropObject, basic_point<LONG>{pt.x, pt.y});
                if (target != nullptr)
                    target->accept(*iActiveInternalDragDropObject);
                iActiveInternalDragDropSource->end_drag_drop(*target);
                iActiveInternalDragDropSource = nullptr;
                iActiveInternalDragDropObject = nullptr;
            }
            return S_OK;
        }
    }
}